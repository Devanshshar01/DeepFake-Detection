"""Enhanced data augmentation utilities for deepfake detection"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_augmentation():
    """Get enhanced augmentation pipeline for training with deepfake-specific augmentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1),
        ], p=0.7),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=7, p=1),
        ], p=0.4),

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
        ], p=0.4),

        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=0.15,
            rotate_limit=20,
            border_mode=0,
            p=0.6
        ),

        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
        ], p=0.3),

        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),

        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),

        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1),
        ], p=0.3),

        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.4),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_augmentation():
    """Get augmentation pipeline for validation"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_time_augmentation():
    """Get test-time augmentation for inference"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
