# encoding:utf-8
"""
Deep-Hough-Transform-Line-Priors (ECCV 2020) https://arxiv.org/abs/2007.09493

Yancong Lin, and Silvia Laura Pintea, and Jan C. van Gemert

e-mail: y.lin-1ATtudelftDOTnl

Vision Lab, Delft University of Technology

MIT license

"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.nn import functional as F


# ####################################HT########################################################
def hough_transform(rows, cols, theta_res, rho_res):
    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))
    theta = theta[0:len(theta) - 1]

    ###  Actually,the offset does not have to be this large, because the origin is located at the image center.
    D = np.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)
    ###  replace the line above to reduce unnecessray computation (significantly).
    # D = np.sqrt((rows/2) ** 2 + (cols/2) ** 2)

    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, int(nrho))

    w = np.size(theta)
    h = np.size(rho)
    cos_value = np.cos(theta * np.pi / 180.0).astype(np.float32)
    sin_value = np.sin(theta * np.pi / 180.0).astype(np.float32)
    sin_cos = np.concatenate((sin_value[None, :], cos_value[None, :]), axis=0)

    ###  This is much more memory-efficient by shifting the coordinate ####
    coords_r, coords_w = np.ones((rows, cols)).nonzero()
    coords = np.concatenate((coords_r[:, None], coords_w[:, None]), axis=1).astype(np.float32)

    coords[:, 0] = rows - coords[:, 0] - rows // 2
    coords[:, 1] = coords[:, 1] + 1 - cols // 2

    vote_map = (coords @ sin_cos).astype(np.float32)

    vote_index = np.zeros((rows * cols, h, w))
    for i in range(rows * cols):
        for j in range(w):
            rhoVal = vote_map[i, j]
            rhoIdx = np.nonzero(np.abs(rho - rhoVal) == np.min(np.abs(rho - rhoVal)))[0]
            vote_map[i, j] = float(rhoIdx[0])
            vote_index[i, rhoIdx[0], j] = 1

    ### remove all-zero lines in the HT maps ####
    vote_rho_idx = vote_index.reshape(rows * cols, h, w).sum(axis=0).sum(axis=1)
    vote_index = vote_index[:, vote_rho_idx > 0.0, :]
    ### update h, since we remove those HT lines without any votes
    ### slightly different from the original paper, the HT size in this script is 182x60.
    h = (vote_rho_idx > 0.0).sum()
    return vote_index.reshape(rows, cols, h, w)


def inverse_hough_transform(h, w, rows, cols, theta_res, rho_res):
    theta = np.linspace(0, 180.0, int(np.ceil(180.0 / theta_res) + 1.0))
    theta = theta[0:len(theta) - 1]

    D = np.sqrt((rows - 1) ** 2 + (cols - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, int(nrho))

    cos_value = np.cos(theta * np.pi / 180.0).astype(np.float32)
    sin_value = np.sin(theta * np.pi / 180.0).astype(np.float32)
    sin_cos = np.concatenate((sin_value[None, :], cos_value[None, :]), axis=0)

    inverse_map = np.zeros((h, w, rows, cols))

    for i in range(h):
        for j in range(w):
            a = cos_value[j]
            b = sin_value[j]
            x0 = a * rho[i]
            y0 = b * rho[i]
            for x in range(rows):
                for y in range(cols):
                    if abs(a * x + b * y - rho[i]) < 1e-5:
                        inverse_map[i, j, x, y] = 1

    return inverse_map


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
def make_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
    ###  no batchnorm layers
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class HT(nn.Module):
    def __init__(self, vote_index):
        super(HT, self).__init__()
        self.r, self.c, self.h, self.w = vote_index.size()
        self.norm = max(self.r, self.c)
        self.vote_index = vote_index.view(self.r * self.c, self.h * self.w)
        self.total = vote_index.sum(0).max()

    def forward(self, image):
        batch, channel, _, _ = image.size()
        image = image.view(batch, channel, -1).view(batch * channel, -1)
        image = F.relu(image)
        HT_map = image @ self.vote_index
        ### normalization ###
        # HT_map = HT_map/self.total
        ### normalized by max(rows, cols)
        HT_map = HT_map / (self.norm)
        HT_map = HT_map.view(batch, channel, -1).view(batch, channel, self.h, self.w)
        return HT_map


class IHT(nn.Module):
    def __init__(self, vote_index):
        super(IHT, self).__init__()
        self.r, self.c, self.h, self.w = vote_index.size()
        self.vote_index = vote_index.view(self.r * self.c, self.h * self.w).t()

    def forward(self, input_HT):
        batch, channel, _, _ = input_HT.size()
        input_HT = F.relu(input_HT)
        input_HT = input_HT.view(batch, channel, self.h * self.w).view(batch * channel, self.h * self.w)
        IHT_map = input_HT @ self.vote_index
        IHT_map = IHT_map.view(batch, channel, self.r * self.c).view(batch, channel, self.r, self.c)
        # return IHT_map/float(self.w)
        return IHT_map


class HTIHT(nn.Module):
    def __init__(self, vote_index, inplanes, outplanes):
        super(HTIHT, self).__init__()

        self.conv1 = nn.Sequential(
            *make_conv_block(inplanes, inplanes, kernel_size=(9, 1), padding=(4, 0), bias=True, groups=inplanes))
        self.conv2 = nn.Sequential(
            *make_conv_block(inplanes, inplanes * 2, kernel_size=(9, 1), padding=(4, 0), bias=True))
        self.conv3 = nn.Sequential(
            *make_conv_block(inplanes * 2, outplanes, kernel_size=(9, 1), padding=(4, 0), bias=True))
        self.conv4 = nn.Sequential(
            *make_conv_block(outplanes, outplanes, kernel_size=(9, 1), padding=(4, 0), bias=True))

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.ht = HT(vote_index)
        self.iht = IHT(vote_index)

        filtersize = 4
        x = np.zeros(shape=((2 * filtersize + 1)))
        x[filtersize] = 1
        z = []
        for _ in range(0, inplanes):
            sigma = np.random.uniform(low=1, high=2.5, size=(1))
            y = ndimage.filters.gaussian_filter(x, sigma=sigma, order=2)
            y = -y / np.sum(np.abs(y))
            z.append(y)
        z = np.stack(z)
        self.conv1[0].weight.data.copy_(torch.from_numpy(z).unsqueeze(1).unsqueeze(3))
        nn.init.kaiming_normal_(self.conv2[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, **kwargs):
        out = self.ht(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.iht(out)
        return out


class DHT_Layer(nn.Module):
    def __init__(self, vote_index, inplanes, outplanes):
        super(DHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        self.ht = HT(vote_index)

    def forward(self, x):
        x = self.fist_conv(x)
        x = self.ht(x)
        return x


class IHT_Layer(nn.Module):
    def __init__(self, vote_index, inplanes, outplanes):
        super(IHT_Layer, self).__init__()
        self.fist_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        self.iht = IHT(vote_index)
        self.convs = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fist_conv(x)
        x = self.iht(x)
        x = self.convs(x)
        return x


class HT2IHT(nn.Module):
    def __init__(self, vote_index, inplanes, outplanes):
        super(HT2IHT, self).__init__()

        self.conv1 = nn.Sequential(
            *make_conv_block(inplanes, inplanes, kernel_size=(9, 1), padding=(4, 0), bias=True, groups=inplanes))
        self.convs = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, 1, 1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(),
            nn.Conv2d(inplanes, inplanes, 3, 1, 1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU()
        )
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.ht = HT(vote_index)
        self.iht = IHT(vote_index)

        filtersize = 4
        x = np.zeros(shape=((2 * filtersize + 1)))
        x[filtersize] = 1
        z = []
        for _ in range(0, inplanes):
            sigma = np.random.uniform(low=1, high=2.5, size=(1))
            y = ndimage.filters.gaussian_filter(x, sigma=sigma, order=2)
            y = -y / np.sum(np.abs(y))
            z.append(y)
        z = np.stack(z)
        self.conv1[0].weight.data.copy_(torch.from_numpy(z).unsqueeze(1).unsqueeze(3))

    def forward(self, x, **kwargs):
        out = self.ht(x)
        out = self.conv1(out)
        out = self.convs(out)
        out = self.iht(out)
        return out


class CAT_HTIHT(nn.Module):

    def __init__(self, vote_index, inplanes, outplanes):
        super(CAT_HTIHT, self).__init__()
        self.htiht = HT2IHT(vote_index, inplanes, outplanes)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inplanes + outplanes, (inplanes + outplanes) * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d((inplanes + outplanes) * 2, inplanes, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        y = self.htiht(x)
        out = self.conv_cat(torch.cat([x, y], dim=1))
        return out


def test():
    from matplotlib import pyplot as plt
    img = np.zeros((120, 100), dtype=np.uint8)
    # 绘制第一条弧线
    cv2.ellipse(img, (50, 75), (30, 30), 0, 0, 180, 255, 1)

    # 绘制第二条弧线
    cv2.ellipse(img, (50, 50), (20, 40), 0, 0, 270, 255, 1)
    cv2.line(img, (0, 100), (100, 100), 100, 1)
    cv2.line(img, (25, 75), (75, 50), 255, 1)
    cv2.line(img, (45, 50), (55, 50), 255, 1)
    # cv2.circle(img, (80, 80), 10, 200, -1)
    # 生成0-30之间的随机整数噪声
    noise = np.random.randint(0, 30, size=(120, 100), dtype=np.uint8)
    # 将图像与噪声相加，并限制值在0-255范围内
    img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    rows, cols = img.shape
    vote_index = hough_transform(rows, cols, theta_res=1, rho_res=1)
    vote_index2 = torch.from_numpy(vote_index).float().contiguous()
    print(vote_index2.shape)
    img_t = torch.from_numpy(img).float().contiguous()
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    HT_map = HT(vote_index2)(img_t)
    ht_g = HT_map.squeeze(0).squeeze(0)
    # print(HT_map.squeeze(0).squeeze(0).shape)
    plt.imshow(ht_g)
    plt.show()
    IHT_map = IHT(vote_index2)(HT_map)
    plt.imshow(IHT_map.squeeze(0).squeeze(0))
    plt.show()

    feat = CAT_HTIHT(vote_index2, inplanes=1, outplanes=1)(img_t)
    plt.imshow(feat.detach().numpy().squeeze(0).squeeze(0))
    plt.show()
    return vote_index2


def built_CAT_HTIHT(sample, backbone, hidden_dim, theta_res, rho_res, device, inplanes, outplanes):
    if isinstance(sample, (list, torch.Tensor)):
        from util.misc import nested_tensor_from_tensor_list
        images = nested_tensor_from_tensor_list(sample)
    features, pos = backbone(images)
    src, _ = features[-1].decompose()
    src = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)(src)
    _, _, h, w = src.shape
    vote_index = hough_transform(h, w, theta_res, rho_res)
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    cat_htiht = CAT_HTIHT(vote_index, inplanes, outplanes)
    return cat_htiht


def build_val_CAT_HTIHT(theta_res, rho_res, inplanes, outplanes):
    w, h = 25, 25
    vote_index = hough_transform(w, h, theta_res, rho_res)
    vote_index = torch.from_numpy(vote_index).float().contiguous()
    cat_htiht = CAT_HTIHT(vote_index, inplanes, outplanes)
    return cat_htiht

# if __name__ == "__main__":
#     ### Default settings: (128, 128, 3, 1)
#     vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
#     rows, cols, h, w = vote_index.shape
#     print('vote_index', vote_index.shape)
#     # sio.savemat('../../vote_index_128_31.mat', {'vote_index': vote_index})
