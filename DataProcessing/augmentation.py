import torch.nn
import torchvision.transforms.functional as F
from torchvision import transforms
from random import randint
import albumentations as A


def pipeline_tranforms():
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])


def albumentation_pixelwise_transforms():
    return A.Compose([
        A.ImageCompression(1, 50, p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01, p=0.1),
        A.RandomSnow(p=0.1),
        A.GlassBlur(p=0.1),
        A.CLAHE(p=0.1)

    ])


def albumentation_mask_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.25),
        A.ElasticTransform(p=0.1),
        A.OpticalDistortion(p=0.1)
    ])


def image_transforms():
    return transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAdjustSharpness(0.9, 0.1),
        transforms.RandomAutocontrast()
    ])


if __name__ == '__main__':
    pass
