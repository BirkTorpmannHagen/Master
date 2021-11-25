import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision.transforms import Normalize
# from model_of_natural_variation.gmcnn.model.net_with_dropout import InpaintingModel_GMCNN_Given_Mask
from DataProcessing.hyperkvasir import KvasirSyntheticDataset
from torch.utils.data.dataloader import DataLoader
from Models.inpainters import SegGenerator


class Inpainter(nn.Module):
    # wrapper around gmcnn
    def __init__(self, path_to_state_dict):
        super(Inpainter, self).__init__()
        self.config = None
        self.model = SegGenerator()
        self.model.load_state_dict(torch.load(path_to_state_dict))

    def forward(self, img, mask, masked_image):
        polyp = self.model(masked_image)
        merged = (1 - mask) * img + (polyp * mask)
        return merged, polyp

    def get_test(self, split="test"):
        for i, (image, mask, masked_image, part, fname) in enumerate(
                DataLoader(KvasirSyntheticDataset("Datasets/HyperKvasir", split="test"),
                           batch_size=4)):
            with torch.no_grad():
                merged, polyp = self.forward(image, mask, masked_image)
                plt.title("Inpainted image")
                plt.imshow(merged[0].T)
                plt.show()
                # plt.savefig(f"model_of_natural_variation/inpaint_examples/{i}")
            break


if __name__ == '__main__':
    inpainter = Inpainter("Predictors/Inpainters/deeplab-generator-550")
    inpainter.get_test()
