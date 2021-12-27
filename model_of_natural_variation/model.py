import matplotlib.pyplot as plt
import torch

from model_of_natural_variation.perturbator import RandomDraw
import albumentations as alb
from model_of_natural_variation.polyp_inpainter import Inpainter
import torch.nn as nn
import numpy as np


class ModelOfNaturalVariation(nn.Module):
    def __init__(self, T0=0):
        super(ModelOfNaturalVariation, self).__init__()
        self.temp = T0
        self.linstep = 0.1
        self.inpainter = Inpainter("Predictors/Inpainters/no-pretrain-deeplab-generator-4990")
        self.perturbator = RandomDraw()
        self.pixelwise_augments, self.geometric_augments = self.get_encoded_transforms()

    def _map_to_range(self, max, min=0):
        # The temperature varies between 0 and 1, where 0 represents no augmentation and 1 represents the maximum augmentation
        # Obviously, having for example a 100% reduction in brightness is not really productive.
        return min + self.temp * (max - min)

    def get_encoded_transforms(self):
        quality_lower = max(100 - int(self._map_to_range(max=100)), 10)
        quality_upper = np.clip(quality_lower + 10, 10, 100)
        pixelwise = alb.Compose([
            alb.ColorJitter(brightness=self._map_to_range(max=0.2),
                            contrast=self._map_to_range(max=0.2),
                            saturation=self._map_to_range(max=0.2),
                            hue=self._map_to_range(max=0.05), p=self.temp),
            alb.GaussNoise(var_limit=self._map_to_range(max=0.01), p=self.temp),
            alb.ImageCompression(quality_lower=quality_lower,
                                 quality_upper=quality_upper,
                                 p=self.temp)
        ]
        )
        geometric = alb.Compose([alb.RandomRotate90(p=self.temp),
                                 alb.Flip(p=self.temp),
                                 alb.OpticalDistortion(distort_limit=self.temp, p=self.temp)])
        return pixelwise, geometric

    def forward(self, image, mask):
        assert len(image.shape) == 4, "Image must be in BxCxHxW format"
        augmented_imgs = torch.zeros_like(image).cuda()
        augmented_masks = torch.zeros_like(mask).cuda()
        for batch_idx in range(image.shape[0]):  # random transforms to every image in the batch
            aug_img = image[batch_idx].squeeze().cpu().numpy().T
            aug_mask = mask[batch_idx].squeeze().cpu().numpy().T
            # if np.random.rand() < self.temp:
            #     # todo integrate inpainting
            #     inpainting_mask = self.perturbator(mask=mask, rad=0.5)
            #     inpaint = ((inpainting_mask - mask) != 0).astype(int)
            # aug_img = self.inpainter(image=image[batch_idx], mask=mask[batch_idx])
            pixelwise = self.pixelwise_augments(image=aug_img)["image"]
            geoms = self.geometric_augments(image=pixelwise, mask=aug_mask)
            augmented_imgs[batch_idx] = torch.Tensor(geoms["image"].T)
            augmented_masks[batch_idx] = torch.Tensor(geoms["mask"].T)
        return augmented_imgs, augmented_masks

    def step(self):
        self.temp = np.clip(self.temp + self.linstep, 0, 1)
        self.pixelwise_augments, self.geometric_augments = self.get_encoded_transforms()


if __name__ == '__main__':
    from DataProcessing.hyperkvasir import KvasirSegmentationDataset
    from torch.utils.data import DataLoader

    mnv = ModelOfNaturalVariation(1)
    for x, y, fname in DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", augment=False)):
        img = x
        mask = y

        mnv(img, mask)
        input()
