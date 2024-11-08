
### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import albumentations as A

### Internal Imports ###

def get_debug_transform():
    transform = A.Compose([
        A.AdvancedBlur(blur_limit=(15, 17), sigmaX_limit=(11.0, 13.0), sigmaY_limit=(11.0, 13.0), p=0.99),
    ])
    return transform

def get_basic_transform():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.6),
        A.Blur(blur_limit=2, p=0.4),
        A.OpticalDistortion(p=0.6),
        A.GridDistortion(p=0.6),
    ])
    return transform

def get_transform_basic_1():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.75, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.5),
        A.Blur(blur_limit=2, p=0.5),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.3),
        A.PixelDropout(p=0.5),
        A.RandomBrightness(p=0.5),
        A.Sharpen(p=0.5),
    ])
    return transform

def get_transform_setup_1():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.75, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.MultiplicativeNoise(p=0.2),
        A.PixelDropout(p=0.5),
        A.RandomBrightness(p=0.5),
        A.Sharpen(p=0.5),

    ])
    return transform

def get_transform_setup_2():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Downscale(scale_min=0.25, scale_max=0.75, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.MultiplicativeNoise(p=0.2),
        A.PixelDropout(p=0.5),
        A.RandomBrightness(p=0.5),
        A.Sharpen(p=0.5),
    ])
    transform = A.RandomOrder(transform)
    return transform



def transform_1():
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.Downscale(scale_min=0.4, scale_max=0.99, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25), p=0.5),
        A.Blur(blur_limit=11, p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.MultiplicativeNoise(p=0.2),
        A.PixelDropout(dropout_prob=0.02, per_channel=False, p=0.5),
        A.PixelDropout(dropout_prob=0.02, per_channel=True, p=0.5),
        A.Sharpen(p=0.5),
    ])
    return transform







def apply_transform(image, transform):
    return transform(image=image)['image']