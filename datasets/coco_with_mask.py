import argparse
import os
import shutil

import cv2
import imgviz
import numpy as np
import tqdm
from PIL import Image
from pycocotools.coco import COCO


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def main(args):
    annotation_file = os.path.join(args.input_dir, 'annotations', 'instances_{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'train2014_annotations'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'Binary_map_aug'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            pre_cat_mask = coco.annToMask(anns[0])
            mask = pre_cat_mask * (catIds.index(anns[0]['category_id']) + 1)
            for i in range(len(anns) - 1):
                binary_mask = coco.annToMask(anns[i + 1])
                mask += binary_mask * (catIds.index(anns[i + 1]['category_id']) + 1)
                mask_area = pre_cat_mask + binary_mask
                mask_intersection_area = np.where(mask_area == 2)
                if len(mask_intersection_area[0]) > 0:
                    mask[mask_intersection_area[0]][mask_intersection_area[1]] = \
                        catIds.index(anns[i + 1]['category_id']) + 1
                    mask_area[mask_intersection_area[0]][mask_intersection_area[1]] = 1
                # for j in range(len(mask_intersection_area[0])):
                #     mask[mask_intersection_area[0][j]][mask_intersection_area[1][j]] = \
                #         catIds.index(anns[i + 1]['category_id']) + 1
                #     mask_area[mask_intersection_area[0][j]][mask_intersection_area[1][j]] = 1
                pre_cat_mask = mask_area
            img_origin_path = os.path.join(args.input_dir, 'images', args.split, img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'train2014_annotations',
                                           img['file_name'].replace('.jpg', '.png'))
            seg_output_path_show = os.path.join(args.input_dir, 'Binary_map_aug',
                                                img['file_name'].replace('.jpg', '.png'))
            if len(np.where(mask > 80)[0]) > 0:
                print("error")
            shutil.copy(img_origin_path, img_output_path)
            cv2.imwrite(seg_output_path, mask)
            save_colored_mask(mask, seg_output_path_show)
    print("process end")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/yc/onionDocument/code/model/MSCOCO2014/", type=str,
                        help="input dataset directory")
    parser.add_argument("--split", default="train2014", type=str,
                        help="train2014 or val2014")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
