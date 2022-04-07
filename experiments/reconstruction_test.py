import matplotlib.pyplot as plt

from models.segmentation_models import *
from data.hyperkvasir import KvasirSegmentationDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import copy


class SplicedReconstructor(nn.Module):
    def __init__(self):
        super(SplicedReconstructor, self).__init__()
        inductivenet = InductiveNet()
        inductivenet.load_state_dict(torch.load("Predictors/Augmented/InductiveNet/consistency_1"))
        self.decoder = copy.deepcopy(inductivenet.reconstruction_decoder)
        self.head = copy.deepcopy(inductivenet.reconstruction_head)
        del inductivenet
        deeplab = DeepLab()
        deeplab.load_state_dict(torch.load("Predictors/Augmented/DeepLab/consistency_1"))
        self.encoder = copy.deepcopy(deeplab.encoder)
        del deeplab

    def predict(self, x):
        features = self.encoder(x)
        reconstructor_output = self.decoder(*features)
        reconstructed = self.head(reconstructor_output)
        return reconstructed


if __name__ == '__main__':
    model = SplicedReconstructor().to("cuda").eval()

    for x, y, _ in DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir/", "test")):
        with torch.no_grad():
            reconstruction = model.predict(x.to("cuda")).cpu()
        fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True, figsize=(2, 1), dpi=1000)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0].imshow(reconstruction[0].T)
        ax[1].imshow(x[0].T)
        plt.show()
        print("Showing...")
