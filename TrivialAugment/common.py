import logging
import warnings
import random
from copy import copy
from typing import Union
from collections import Counter

import numpy as np
import torch
from torch.utils.checkpoint import check_backward_validity, detach_variable, get_device_states, set_device_states
from torchvision.datasets import VisionDataset, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import Subset, ConcatDataset

from PIL import Image

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def copy_and_replace_transform(ds: Union[CIFAR10, ImageFolder, Subset], transform):
    assert ds.dataset.transform is not None if isinstance(ds,Subset) else (all(d.transform is not None for d in ds.datasets) if isinstance(ds,ConcatDataset) else ds.transform is not None) # make sure still uses old style transform
    if isinstance(ds, Subset):
        new_super_ds = copy(ds.dataset)
        new_super_ds.transform = transform
        new_ds = copy(ds)
        new_ds.dataset = new_super_ds
    elif isinstance(ds, ConcatDataset):
        def copy_and_replace_transform(ds):
            new_ds = copy(ds)
            new_ds.transform = transform
            return new_ds

        new_ds = ConcatDataset([copy_and_replace_transform(d) for d in ds.datasets])

    else:
        new_ds = copy(ds)
        new_ds.transform = transform
    return new_ds

def apply_weightnorm(nn):
    def apply_weightnorm_(module):
        if 'Linear' in type(module).__name__ or 'Conv' in type(module).__name__:
            torch.nn.utils.weight_norm(module, name='weight', dim=0)
    nn.apply(apply_weightnorm_)


def shufflelist_with_seed(lis, seed='2020'):
    s = random.getstate()
    random.seed(seed)
    random.shuffle(lis)
    random.setstate(s)


def stratified_split(labels, val_share):
    assert isinstance(labels, list)
    counter = Counter(labels)
    indices_per_label = {label: [i for i,l in enumerate(labels) if l == label] for label in counter}
    per_label_split = {}
    for label, count in counter.items():
        indices = indices_per_label[label]
        assert count == len(indices)
        shufflelist_with_seed(indices, f'2020_{label}_{count}')
        train_val_border = round(count*(1.-val_share))
        per_label_split[label] = (indices[:train_val_border], indices[train_val_border:])
    final_split = ([],[])
    for label, split in per_label_split.items():
        for f_s, s in zip(final_split, split):
            f_s.extend(s)
    shufflelist_with_seed(final_split[0], '2020_yoyo')
    shufflelist_with_seed(final_split[1], '2020_yo')
    return final_split


def denormalize(img, mean, std):
    mean, std = torch.tensor(mean).to(img.device), torch.tensor(std).to(img.device)
    return img.mul_(std[:,None,None]).add_(mean[:,None,None])

def normalize(img, mean, std):
    mean, std = torch.tensor(mean).to(img.device), torch.tensor(std).to(img.device)
    return img.sub_(mean[:,None,None]).div_(std[:,None,None])