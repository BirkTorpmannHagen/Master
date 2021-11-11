import torch.nn as nn
import torch
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus


class SegGenerator(nn.Module):
    def __init__(self):
        super(SegGenerator, self).__init__()
        self.model = DeepLabV3Plus(in_channels=3, classes=3, activation=None, encoder_weights=None)

    def forward(self, mask):
        return self.model(mask)


class SegDiscriminator(nn.Module):
    def __init__(self):
        super(SegDiscriminator, self).__init__()
        self.model = DeepLabV3Plus(in_channels=3, classes=1, activation="sigmoid", encoder_weights=None)

    def forward(self, mask):
        # adds nosie to inputs, see https://arxiv.org/pdf/1701.04862.pdf
        return self.model(mask + torch.normal(torch.zeros_like(mask), torch.ones_like(mask) / 10))
        # return self.model(mask)
