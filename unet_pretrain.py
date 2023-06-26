import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from models.backbone import build_backbone
from models.unet.TransUNet import TransUNet
from models.unet.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--distributed', action='store_true', default=False, help='Distributed training')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Number of hidden channels')
    parser.add_argument('--num_class', type=int, default=1, help='Number of classes')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay')
    parser.add_argument('--dir_checkpoint', type=str, help='save weights')
    parser.add_argument('--coco_path', type=str, default=False, help='coco_path')
    return parser.parse_args()


def build_dataset_coco(args):
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    return data_loader_train, data_loader_val


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for samples, targets in dataloader:
            samples = samples.tensors.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = [{k: v for k, v in t.items()} for t in targets]
            masks = [t["masks"] for t in targets]
            mask_true = torch.stack([tensor.sum(dim=0, keepdim=True) for tensor in masks], dim=0) \
                .bool() \
                .squeeze_(1) \
                .to(device=device, dtype=torch.long)
            mask_pred = model(samples).squeeze(1)
            if model.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


def init(args):
    args.masks = True
    args.dataset_file = 'coco_powerline'
    args.distributed = False
    args.num_workers = 2
    args.position_embedding = 'sine'
    args.hidden_dim = 256
    args.backbone = 'resnet34'
    args.lr_backbone = 1e-5
    args.dilation = False
    args.epochs = 5
    args.amp = True
    args.num_class = 1
    # args.batch_size = 1
    # args.lr = 1e-4
    # args.coco_path = 'D:\dataset\coco_powerline_1'
    args.dir_checkpoint = './output'
    return args


if __name__ == '__main__':
    args = get_args()
    args = init(args)
    in_channels = args.hidden_dim
    out_channels = in_channels
    n_classes = args.num_class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 　model = UNet(in_channels, out_channels, n_classes, bilinear=False)

    backbone = build_backbone(args)
    backbone[0].body.maxpool = backbone[0].body.relu
    model = TransUNet(backbone,
                      num_classes=n_classes,
                      in_channels=512,
                      d_model=args.hidden_dim)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.in_channels} input channels\n'
                 f'\t{model.out_channels} output channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    data_loader_train, data_loader_val = build_dataset_coco(args)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(),
                              lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.999)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global_step = 0
    # sample = next(iter(data_loader_train))
    # a, b = sample
    # prediction = model(a.tensors.to(device))

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
             val_percent=0.1, save_checkpoint=True, img_scale=args.scale, amp=args.amp)
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        prin_freq = 10
        for samples, targets in data_loader_train:
            samples = samples.tensors.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            targets = [{k: v for k, v in t.items()} for t in targets]
            masks = [t["masks"] for t in targets]
            # 处理coco实例分割掩码
            # 实例mask累加-->语义mask
            true_masks = torch.stack([tensor.sum(dim=0, keepdim=True) for tensor in masks], dim=0) \
                .bool()\
                .squeeze(1)\
                .to(device=device, dtype=torch.long)
            # 二分类 one-hot编码
            # true_masks = F.one_hot(true_masks.squeeze_(1), 2).permute(0, 3, 1, 2).float()
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                masks_pred = model(samples).squeeze(1)
                if model.n_classes == 1:
                    loss = criterion(masks_pred, true_masks.float())
                    loss += dice_loss(F.sigmoid(masks_pred), true_masks.float(), multiclass=False)
                else:
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(), true_masks,
                        multiclass=True
                    )
            if (len(data_loader_train) * args.batch_size) % (
                    (len(data_loader_train) * args.batch_size) // prin_freq) == 0:
                print(f"loss:{loss}")
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })

            # Evaluation round
            division_step = len(data_loader_train) * args.batch_size

            if global_step % division_step == 0:
                histograms = {}
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('/', '.')
                #     if not (torch.isinf(value) | torch.isnan(value)).any():
                #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                val_score = evaluate(model, data_loader_val, device, args.amp)
                scheduler.step(val_score)

                logging.info('Validation Dice score: {}'.format(val_score))
                try:
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'images': wandb.Image(samples[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })
                except:
                    pass

        Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        checkpoint_path = os.path.join(args.dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch))
        torch.save(state_dict, checkpoint_path)
        logging.info(f'Checkpoint {epoch} saved!')
