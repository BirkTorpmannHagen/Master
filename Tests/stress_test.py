import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AdditiveNoise(nn.Module):
    def __init__(self, noise_factor):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, img):
        return img + torch.randn(img.shape) * self.noise_factor

    def __repr__(self):
        return self.__class__.__name__ + '(noise_factor={})'.format(self.noise_factor)


def apply_stressors(image, mask):
    augs = transforms.Compose([AdditiveNoise(5),
                               transforms.])
