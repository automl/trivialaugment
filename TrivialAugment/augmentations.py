# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py

import numpy as np
import torch

from TrivialAugment import autoaugment, fast_autoaugment
import aug_lib


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_randaugment(n,m,weights,bs):
    if n == 101 and m == 101:
        return autoaugment.CifarAutoAugment(fixed_posterize=False)
    if n == 102 and m == 102:
        return autoaugment.CifarAutoAugment(fixed_posterize=True)
    if n == 201 and m == 201:
        return autoaugment.SVHNAutoAugment(fixed_posterize=False)
    if n == 202 and m == 202:
        return autoaugment.SVHNAutoAugment(fixed_posterize=False)
    if n == 301 and m == 301:
        return fast_autoaugment.cifar10_faa
    if n == 401 and m == 401:
        return fast_autoaugment.svhn_faa
    assert m < 100 and n < 100
    if m == 0:
        if weights is not None:
            return aug_lib.UniAugmentWeighted(n, probs=weights)
        elif n == 0:
            return aug_lib.UniAugment()
        else:
            raise ValueError('Wrong RandAug Params.')
    else:
        assert n > 0 and m > 0
        return aug_lib.RandAugment(n, m)
