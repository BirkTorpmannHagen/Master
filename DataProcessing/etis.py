from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from PIL.Image import open
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import DataProcessing.augmentation as aug


class EtisDataset(Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "ETIS-LaribPolypDB")))
        self.common_transforms = aug.pipeline_tranforms()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(self.path, "ETIS-LaribPolypDB/{}.jpg".format(index + 1))).convert("RGB"))
        mask = self.common_transforms(
            open(join(self.path, "GroundTruth/p{}.jpg".format(index + 1))).convert("RGB"))
        mask = (mask > 0.5).float()
        return image, mask, index + 1


def test_etis():
    for x, y, in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
        plt.imshow(x)
        plt.show()
