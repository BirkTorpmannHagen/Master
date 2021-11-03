import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision.transforms import Normalize
# from PipelineMods.gmcnn.model.net_with_dropout import InpaintingModel_GMCNN_Given_Mask
from PipelineMods.ganlib.implementations.context_encoder.models import Generator
from DataProcessing.hyperkvasir import KvasirSyntheticDataset
from torch.utils.data.dataloader import DataLoader


class Inpainter(nn.Module):
    # wrapper around gmcnn
    def __init__(self, path_to_state_dict):
        super(Inpainter, self).__init__()
        self.config = None
        self.model = Generator(3)
        self.model.load_state_dict(torch.load(path_to_state_dict))

    def forward(self, img, mask):
        polyp = self.model.forward(mask)

        merged = (1 - mask) * img + (polyp * mask)
        return merged, polyp


if __name__ == '__main__':
    inpainter = Inpainter("Predictors/Inpainters/generator-190")
    for image, mask, masked_image, part, fname in DataLoader(KvasirSyntheticDataset("Datasets/HyperKvasir")):
        with torch.no_grad():
            merged, polyp = inpainter(image, mask)
            # print(torch.max(mask))
            # plt.imshow((polyp[0] * (mask))[0].T)
            # plt.show()
            # plt.imshow(masked_image[0].T)
            plt.imshow(merged[0].T)
            plt.show()
