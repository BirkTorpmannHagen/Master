import matplotlib.pyplot as plt

from models.segmentation_models import DeepLab
from data.hyperkvasir import KvasirSyntheticDataset
import torch
from torch.utils.data import DataLoader


def test():
    model = DeepLab().to("cuda")
    model.load_state_dict(torch.load("Predictors/Augmented/DeepLab/inpainter_augmentation_1"))
    for x, y, _ in DataLoader(KvasirSyntheticDataset("Datasets/HyperKvasir")):
        x = x.to("cuda")
        y = y.to("cuda")
        out = model.predict(x)
        plt.imshow(x[0].cpu().T)
        plt.axis("off")
        plt.show(bbox_inches='tight', pad_inches=0)
        plt.imshow(y[0].cpu().T, alpha=0.5)
        plt.show()


if __name__ == '__main__':
    test()
