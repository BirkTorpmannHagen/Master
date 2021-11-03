from os import listdir
from os.path import join

import numpy as np
import torch.utils.data
from PIL.Image import open
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms
from utils.mask_generator import generate_a_mask


class KvasirClassificationDataset(Dataset):
    """
    Dataset class that fetches images with the associated pathological class labels for use in Pretraining
    """

    def __init__(self, path):
        super(KvasirClassificationDataset, self).__init__()
        self.path = join(path, "labeled-images/lower-gi-tract/pathological-findings")
        self.label_names = listdir(self.path)
        self.num_classes = len(self.label_names)
        self.fname_class_dict = {}
        i = 0
        self.class_weights = np.zeros(self.num_classes)
        for i, label in enumerate(self.label_names):
            class_path = join(self.path, label)
            for fname in listdir(class_path):
                self.class_weights[i] += 1
                self.fname_class_dict[fname] = label
        self.index_dict = dict(zip(self.label_names, range(self.num_classes)))

        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])

    def __len__(self):
        # return 256  # for debugging
        return len(self.fname_class_dict)

    def __getitem__(self, item):
        fname, label = list(self.fname_class_dict.items())[item]
        onehot = one_hot(torch.tensor(self.index_dict[label]), num_classes=self.num_classes)
        image = open(join(join(self.path, label), fname)).convert("RGB")
        # print(list(image.getdata()))
        # input()
        image = self.common_transforms(open(join(join(self.path, label), fname)).convert("RGB"))
        return image, onehot.float(), fname


class KvasirSegmentationDataset(Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
        Employs "vanilla" augmentations,
    """

    def __init__(self, path):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = self.common_transforms(
            open(join(join(self.path, "masks/"), self.fnames[index])).convert("L"))
        mask = (mask > 0.5).float()
        return image, mask, self.fnames[index]


class KvasirInpaintingDataset(Dataset):
    def __init__(self, path):
        super(KvasirInpaintingDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = self.common_transforms(
            open(join(join(self.path, "masks/"), self.fnames[index])).convert("L"))
        mask = (mask > 0.5).float()

        part = mask * image
        masked_image = image - part

        return image, mask, masked_image, part, self.fnames[index]


class KvasirSyntheticDataset(Dataset):
    def __init__(self, path):
        super(KvasirSyntheticDataset, self).__init__()
        self.path = join(path, "unlabeled-images")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = generate_a_mask(imsize=400).T
        mask = torch.tensor(mask > 0.5).float()
        part = mask * image
        masked_image = image - part

        return image, mask, masked_image, part, self.fnames[index]


def test_KvasirSegmentationDataset():
    dataset = KvasirSegmentationDataset("Datasets/HyperKvasir")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    print("Classification Tests passed")


def test_KvasirClassificationDataset():
    dataset = KvasirClassificationDataset("Datasets/HyperKvasir")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    print("Segmentation Tests passed")


if __name__ == '__main__':
    test_KvasirSegmentationDataset()
    test_KvasirClassificationDataset()
