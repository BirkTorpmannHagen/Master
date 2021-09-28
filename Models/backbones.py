import segmentation_models_pytorch as backbones


class Unet(backbones.Unet):
    def __init__(self, classes):
        super(Unet, self).__init__()
        self.model = backbones.Unet(in_channels=3, classes=classes)

    def forward(self, x):
        return self.model.forward(x)


class DeepLab(backbones.DeepLabV3Plus):
    def __init__(self, classes):
        super(DeepLab, self).__init__()
        self.model = backbones.DeepLabV3Plus(in_channels=3, classes=classes, encoder_weights=None)

    def forward(self, x):
        return self.model.forward(x)


