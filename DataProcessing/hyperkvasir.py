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
        self.common_transforms = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((400, 400)),
                                                     ])
        self.train_transforms = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                    ])

    def __len__(self):
        return len(self.fnames)

    def get_augmentations(self):
        pass

    def __getitem__(self, index):
        image = (open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = (open(join(join(self.path, "masks/"), self.fnames[index])).convert("RGB"))
        image = self.common_transforms(image)
        # image = self.train_transforms(image)
        mask = self.common_transforms(mask)[0].unsqueeze(0)
        mask = (mask > 0.5).float()
        return image, mask, self.fnames[index]


if __name__ == '__main__':
    dataset = KvasirDataset("Data/segmented-images/")
    image, mask = dataset[1]
    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()
