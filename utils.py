# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import numpy as np



def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data

            
def get_tiny_imagenet(args):
    # copied from GibbsNet_pytorch/load.py
    workers = 0 if args.debug else 4
    batch_size = args.batch_size

    mean = [x / 255 for x in [127.5, 127.5, 127.5]]
    std = [x / 255 for x in [127.5, 127.5, 127.5]]

    train_transform = tr.Compose(
        [tr.RandomHorizontalFlip(),
         tr.RandomCrop(64, padding=4),
         tr.ToTensor(),
         tr.Normalize(mean, std)])

    test_transform = tr.Compose(
        [tr.ToTensor(), tr.Normalize(mean, std)])
    train_root = os.path.join(args.data_root, 'train')  # this is path to training images folder
    validation_root = os.path.join(args.data_root, 'val/images')  # this is path to validation images folder
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(validation_root, transform=test_transform)
    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    temp2 = cycle(labelled)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    validation = test

    return labelled, temp2, validation, test


def get_data(args):
    if args.dataset == 'tinyimagenet':
        return get_tiny_imagenet(args)
    elif args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5))]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5))]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True, split="train" if train else "test")
    
    num_workers = 0 if args.debug else 4
    dset_train = dataset_fn(True, transform_train)
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dset_test = dataset_fn(False, transform_test)
    test_loader = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def isnan(tensor):
    return (tensor != tensor)


def tens2numpy(tens):
    if tens.is_cuda:
        tens = tens.cpu()
    if tens.requires_grad:
        tens = tens.detach()
    return tens.numpy()


def t2n(tens):
    if isinstance(tens, np.ndarray):
        return tens
    elif isinstance(tens, list):
        return np.array(tens)
    elif isinstance(tens, float) or isinstance(tens, int):
        return np.array([tens])
    else:
        return tens2numpy(tens)


def n2t(tens):
    return torch.from_numpy(tens)


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()


def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


def plot_curve_peak(x, h, label, title='', mini=False, save_path=None):
    import matplotlib.pyplot as plt
    if mini:
        peak = np.argmin(h)
    else:
        peak = np.argmax(h)
    peak_v = h[peak]

    plt.plot(x, h, label=label)
    plt.plot(x[peak], h[peak], 'o', color='red')
    plt.annotate('%.2f' % peak_v, xy=(x[peak], h[peak]), xytext=(x[peak], h[peak]))
    plt.plot(x[-1], h[-1], 'o', color='red')
    plt.annotate('%.2f' % h[-1], xy=(x[-1], h[-1]), xytext=(x[-1], h[-1]))
    plt.title(title)
    plt.ylabel(label)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()
