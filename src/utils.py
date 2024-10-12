import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import sys

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator


# create dataset with power transmittance, wavelength conditions
class PowerWaveDataset(Dataset):
    def __init__(
        self,
        design_folder,
        wave_path,
        power_path,
        transform=T.ToTensor(),
        normalize=True,
    ):
        self.design_folder = design_folder
        self.wave_data = np.loadtxt(wave_path, delimiter=",", skiprows=1)
        self.power_data = np.loadtxt(power_path, delimiter=",", skiprows=1)
        self.transform = transform
        self.normalize = normalize
        self.samples = []
        self.labels = []

        # Load design images
        design_files = []
        for file in os.listdir(design_folder):
            if file.endswith(".tiff"):
                design_files.append(file)
        design_files.sort()
        self.designs = design_files  # List of design file names(str): ['design1.tiff', 'design2.tiff', ...]

        # Now, create a list of all combinations
        for idx, design_file in enumerate(self.designs):
            design_path = os.path.join(self.design_folder, design_file)
            design_sample = Image.open(design_path).convert("L")
            design_sample = self.transform(
                design_sample
            )  # transform grayscale image to tensor [0, 1]
            if self.normalize:
                design_sample = (design_sample - 0.5) / 0.5  # [0, 1] -> [-1, 1]

            # For each of the 24 power and wavelength values
            power_samples = self.power_data[idx, :]  # shape (24,): row of power values
            wave_samples = self.wave_data[idx, :]  # shape (24,): row of wave values

            for power, wave in zip(power_samples, wave_samples):
                power_ratio = (
                    power / 240000
                )  # normalized power value = power trasmittance / 240000
                normalized_wave = (wave - 400) / 1200  # normalized wave value
                label = np.array([power_ratio, normalized_wave], dtype=np.float32)

                self.samples.append((design_sample, torch.from_numpy(label)))
                # [(design_sample1, (power_ratio1_1, normalized_wave1_1)), ... , (design_sample1, (power_ratio1_24, normalized_wave1_24)), (design_sample2, (power_ratio2_1, normalized_wave2_1)), ... (design_sample24, (power_ratio24_24, normalized_wave24_24))]
                self.labels.append(label)
                # [(power_ratio1_1, normalized_wave1_1), ... , (power_ratio1_24, normalized_wave1_24), (power_ratio2_1, normalized_wave2_1), ... (power_ratio24_24, normalized_wave24_24)]

        # Convert python list to numpy array
        self.labels = np.array(self.labels)  # shape: (num_samples, 2)

    def __len__(self):
        return len(self.samples)  # 24 * num_designs

    def __getitem__(self, idx):  # idx: 0 ~ [24 * (num_designs - 1)]
        design_sample, label = self.samples[idx]
        return {"design": design_sample, "labels": label}


# helper functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def divisible_by(numer, denom):
    return (numer % denom) == 0


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    """
    return [divisor, divisor, ..., remainder] that sum up to num
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions
def normalize_to_neg_one_to_one(img):
    """
    Normalize image from [0, 1] to [-1, 1]
    """
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """ "
    Unnormalize image from [-1, 1] to [0, 1]
    """
    return (t + 1) * 0.5


# number of parameters
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


# Progress Bar
class SimpleProgressBar:
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100  # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x):
            return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write(
            "\r%d%% [%s]" % (int(x), "#" * pointer + "." * (self.width - pointer))
        )
        sys.stdout.flush()
        if x == 100:
            print("")


## dataset
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception(
                    "images ("
                    + str(len(self.images))
                    + ") and labels ("
                    + str(len(self.labels))
                    + ") do not have the same length!!!"
                )
        self.normalize = normalize

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image / 255.0
            image = (image - 0.5) / 0.5  # ? to [0,1] or [-1,1]

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


# compute entropy of class labels; labels is a numpy array
def compute_entropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = np.e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


def predict_class_labels(net, images, batch_size=500, verbose=False, num_workers=0):
    net = net.cuda()
    net.eval()

    n = len(images)
    if batch_size > n:
        batch_size = n
    dataset_pred = IMGs_dataset(images, normalize=False)
    dataloader_pred = torch.utils.data.DataLoader(
        dataset_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_labels_pred = np.zeros(n + batch_size)
    with torch.no_grad():
        nimgs_got = 0
        if verbose:
            pb = SimpleProgressBar()
        for batch_idx, batch_images in enumerate(dataloader_pred):
            batch_images = batch_images.type(torch.float).cuda()
            batch_size_curr = len(batch_images)

            outputs, _ = net(batch_images)
            _, batch_class_labels_pred = torch.max(outputs.data, 1)
            class_labels_pred[nimgs_got : (nimgs_got + batch_size_curr)] = (
                batch_class_labels_pred.detach().cpu().numpy().reshape(-1)
            )

            nimgs_got += batch_size_curr
            if verbose:
                pb.update((float(nimgs_got) / n) * 100)
        # end for batch_idx
    class_labels_pred = class_labels_pred[0:n]
    return class_labels_pred


# horizontal flip images (tensor version)
def hflip_images_tensor(batch_images):
    """배치 내의 텐서 이미지를 수평으로 플립합니다."""
    uniform_threshold = torch.rand(len(batch_images))
    indx_gt = torch.where(uniform_threshold > 0.5)[0]
    batch_images[indx_gt] = torch.flip(
        batch_images[indx_gt], dims=[3]
    )  # 너비 차원 플립
    return batch_images


# normalize images
def normalize_images(batch_images, to_neg_one_to_one=False):
    if to_neg_one_to_one:
        batch_images = (batch_images - 0.5) / 0.5
    return batch_images
