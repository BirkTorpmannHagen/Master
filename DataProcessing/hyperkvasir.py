from os import listdir
from os.path import join

import matplotlib.pyplot as plt
from PIL.Image import open
from torch.utils.data import Dataset
from torchvision import transforms


class KvasirDataset(Dataset):
    def __init__(self, path):
        super(KvasirDataset, self).__init__()
        self.path = path
        self.fnames = listdir(join(path, "images/"))
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.CenterCrop(400)
                                              ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = (open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = (open(join(join(self.path, "masks/"), self.fnames[index])).convert("RGB"))
        image = self.transforms(image)
        mask = self.transforms(mask)
        return image, mask, self.fnames[index]


if __name__ == '__main__':
    dataset = KvasirDataset("Data/segmented-images/")
    image, mask = dataset[1]
    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()
