import segmentation_models_pytorch as backbones
import torch.nn as nn

"""
Model wrappers
"""


class Unet(backbones.Unet):
    def __init__(self):
        super(Unet, self).__init__()
        self.model = backbones.Unet(in_channels=3, classes=1)

    def forward(self, x):
        return self.model.forward(x)


class DeepLab(backbones.DeepLabV3Plus):
    def __init__(self, encoder_weights=None):
        super(DeepLab, self).__init__()
        self.model = backbones.DeepLabV3Plus(in_channels=3, classes=1, encoder_weights=encoder_weights,
                                             activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)


class DivergentNet(nn.Module):
    def __init__(self):
        super(DivergentNet, self).__init__()

    def forward(self, x):
        pass


class PolypNet(nn.Module):
    def __init__(self):
        super(PolypNet, self).__init__()

    def forward(self, x):
        pass
