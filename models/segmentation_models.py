import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base as smp_base
import torch.nn as nn
import torch

"""
Model wrappers
"""


class Unet(smp.Unet):
    def __init__(self):
        super(Unet, self).__init__()
        self.model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class FPN(smp.fpn.FPN):
    def __init__(self):
        super(FPN, self).__init__()
        self.model = smp.Unet(in_channels=3, classes=1, activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()|
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class DeepLab(smp.DeepLabV3Plus):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.model = smp.DeepLabV3Plus(in_channels=3, classes=1,
                                       activation="sigmoid")

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


"""
models
"""


class TriUnet(nn.Module):
    def __init__(self):
        super(TriUnet, self).__init__()
        self.Unet1 = Unet()
        self.Unet2 = Unet()
        self.Unet3 = smp.Unet(in_channels=2, classes=1, activation="sigmoid")
        self.deeplab = DeepLab()
        self.FPN = FPN()

    def forward(self, x):
        mask1, mask2 = self.Unet1(x), self.Unet2(x)
        mask3 = self.Unet3(torch.cat((mask1, mask2), 1))
        return mask3

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask = self.forward(x)
        return mask


class InductiveNet(smp.DeepLabV3Plus):
    """
    Modified Deeplab with auxilliary task and extra decoder
    """

    def __init__(self, in_channels=3):
        super(InductiveNet, self).__init__(in_channels=in_channels, classes=1,
                                           activation="sigmoid")
        self.reconstruction_decoder = smp.deeplabv3.model.DeepLabV3PlusDecoder(self.encoder.out_channels)
        self.reconstruction_head = smp.base.SegmentationHead(self.reconstruction_decoder.out_channels, 3, kernel_size=1,
                                                             upsampling=4)

    def forward(self, x):
        # return super(InductiveNet, self).forward(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        reconstructor_output = self.reconstruction_decoder(*features)
        masks = self.segmentation_head(decoder_output)
        reconstructed = self.reconstruction_head(reconstructor_output)
        return masks, reconstructed

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            mask, _ = self.forward(x)
        return mask
