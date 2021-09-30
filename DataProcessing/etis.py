from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from PIL.Image import open
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EtisDataset(Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "ETIS-LaribPolypDB")))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(self.path, "ETIS-LaribPolypDB/{}.jpg".format(index))).convert("RGB"))
        mask = self.common_transforms(
            open(join(self.path, "GroundTruth/p{}.jpg".format(index))).convert("RGB"))
        mask = (mask > 0.5).float()
        return image, mask


def test_etis():
    for x, y, in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
        plt.imshow(x)
        plt.show()