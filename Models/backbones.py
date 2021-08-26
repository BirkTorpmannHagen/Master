import segmentation_models_pytorch as backbones


class Unet(backbones.Unet):
    def __init__(self, classes):
        super(Unet, self).__init__()
        self.model = backbones.Unet(in_channels=3, classes=classes)

    def forward(self, x):
        return self.model.forward(x)
