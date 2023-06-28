import matplotlib.pyplot as plt
import torch
from torch import nn

from models.fpn.fpn import FPN18, FPN50, FPN101, ResNext50_FPN


class Net(nn.Module):
    def __init__(self, numAngle, numRho, backbone):
        super(Net, self).__init__()
        if backbone == 'resnet18':
            self.backbone = FPN18(pretrained=True, output_stride=32)

        if backbone == 'resnet50':
            self.backbone = FPN50(pretrained=True, output_stride=16)

        if backbone == 'resnet101':
            self.backbone = FPN101(output_stride=16)

        if backbone == 'resnext50':
            self.backbone = ResNext50_FPN(output_stride=16)

        self.numAngle = numAngle
        self.numRho = numRho

    def upsample_cat(self, p1, p2, p3, p4):
        p1 = nn.functional.interpolate(p1, size=(self.numAngle, self.numRho), mode='bilinear')
        p2 = nn.functional.interpolate(p2, size=(self.numAngle, self.numRho), mode='bilinear')
        p3 = nn.functional.interpolate(p3, size=(self.numAngle, self.numRho), mode='bilinear')
        p4 = nn.functional.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear')
        return torch.cat([p1, p2, p3, p4], dim=1)

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
        return p1, p2, p3, p4


if __name__ == '__main__':
    net = Net(100, 100, 'resnet50')
    weight_path = 'D:/下载项的原始目录/chrome的下载项/dht_r50_fpn_sel-c9a29d40.pth'
    net.load_state_dict(torch.load(weight_path), strict=False)
    image_path = 'D:\dataset/NKL/NKL/1.jpg'
    from PIL import Image

    img = Image.open(image_path)
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(250),
        T.ToTensor(),
    ])
    x = transform(img).unsqueeze(0)
    print(x.shape)
    output = net(x)
    for out in output:
        print(out.size())
        plt.imshow(out[0][25].detach().numpy())
        plt.show()
