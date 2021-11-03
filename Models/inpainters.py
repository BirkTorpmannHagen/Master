import torch.nn as nn
from segmentation_models_pytorch.deeplabv3 import DeepLabV3Plus


class SegGenerator(nn.Module):
    def __init__(self):
        super(SegGenerator, self).__init__()
        self.model = DeepLabV3Plus(in_channels=3, classes=3, activation=None)

    def forward(self, mask):
        return self.model(mask)


class SegDiscriminator(nn.Module):
    def __init__(self):
        super(SegDiscriminator, self).__init__()
        self.model = DeepLabV3Plus(in_channels=3, classes=1, activation="sigmoid")

    def forward(self, mask):
        return self.model(mask)
