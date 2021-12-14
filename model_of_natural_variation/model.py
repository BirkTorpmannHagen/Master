import matplotlib.pyplot as plt

from perturbator import RandomDraw
import albumentations as alb
from polyp_inpainter import Inpainter
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
        pixelwise = alb.Compose([
            alb.ColorJitter(brightness=self._map_to_range(max=0.2),
                            contrast=self._map_to_range(max=0.2),
                            saturation=self._map_to_range(max=0.2),
                            hue=self._map_to_range(max=0.05), p=self.temp),
            alb.GaussNoise(var_limit=self._map_to_range(max=0.01), p=self.temp),
            alb.ImageCompression(quality_lower=max(100 - self._map_to_range(max=100), 10),
                                 quality_upper=min(100 - self._map_to_range(max=100) + 10, 100),
                                 p=self.temp)
        ]
        )
        geometric = alb.Compose([alb.RandomRotate90(p=self.temp),
                                 alb.Flip(p=self.temp),
                                 alb.OpticalDistortion(distort_limit=self.temp, p=self.temp)])
        return pixelwise, geometric

    def forward(self, image, mask):
        for batch_idx in range(image.shape[0]):  # random transforms to every image in the batch
            aug_img = image[batch_idx].squeeze().numpy().T
            aug_mask = mask[batch_idx].squeeze().numpy().T
            pixelwise = self.pixelwise_augments(image=aug_img)["image"]
            plt.imshow(aug_img)
            plt.title("original")
            plt.show()
            # input("continue?")
            # plt.imshow(pixelwise)
            # plt.title("Pixelwise")
            # plt.show()
            geoms = self.geometric_augments(image=pixelwise, mask=aug_mask)
            geom_image = geoms["image"]
            geom_mask = geoms["mask"]
            plt.imshow(geom_image)
            # plt.imshow(geom_mask, alpha=0.25)
            plt.title("geom")
            plt.show()
            if np.random.rand() < self.temp:
                # todo make sure that the inpainting mask only covers the relevant parts
                inpainting_mask = self.perturbator.forward(mask=geom_mask, rad=0.5)
                self.inpainter(image=geom_image, mask=)

    def step(self):
        self.temp = self.temp + self.linstep
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
