from os import listdir
from os.path import join

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL.Image import open
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms
from model_of_natural_variation.model import ModelOfNaturalVariation
import DataProcessing.augmentation as aug
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
        Employs "vanilla" augmentations
    """

    def __init__(self, path, split="train", augment=True):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = aug.pipeline_tranforms()
        self.pixeltrans = aug.albumentation_pixelwise_transforms()
        self.segtrans = aug.albumentation_pixelwise_transforms()
        # deterministic partition
        self.split = split
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.augment = augment
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        image = np.array(open(join(join(self.path, "images/"), self.split_fnames[index])).convert("RGB"))
        mask = np.array(open(join(join(self.path, "masks/"), self.split_fnames[index])).convert("L"))
        if self.split == "train" and self.augment == True:
            transformed = self.pixeltrans(image=image)
            image = transformed["image"]
            segtransformed = self.segtrans(image=image, mask=mask)
            image, mask = segtransformed["image"], segtransformed["mask"]
        image = self.common_transforms(PIL.Image.fromarray(image))
        mask = self.common_transforms(PIL.Image.fromarray(mask))
        mask = (mask > 0.5).float()
        return image, mask, self.split_fnames[index]


class KvasirMNVset(KvasirSegmentationDataset):
    def __init__(self, path, split):
        super(KvasirMNVset, self).__init__(path, split, augment=False)
        self.mnv = ModelOfNaturalVariation(1)
        self.p = 0.5

    def __getitem__(self, index):
        image = np.array(open(join(self.path, "images/", self.split_fnames[index]).convert("RGB")))
        mask = np.array(open(join(self.path, "masks/", self.split_fnames[index]).convert("L")))
        image = self.common_transforms(PIL.Image.fromarray(image))
        mask = self.common_transforms(PIL.Image.fromarray(mask))
        mask = (mask > 0.5).float()
        flag = False
        if self.split == "train" and np.random.rand() < self.p:
            flag = True
            image, mask = self.mnv(image.unsqueeze(0), mask.unsqueeze(0))
            image = image.squeeze()
            mask = mask.squeeze(0)  # todo make this less ugly
        return image, mask, self.split_fnames[index], flag

    def set_prob(self, prob):
        self.p = prob


class KvasirInpaintingDataset(Dataset):
    def __init__(self, path, split="train"):
        super(KvasirInpaintingDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])
        self.split = split
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")

    def __len__(self):
        return len(self.split_fnames)

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(join(self.path, "images/"), self.split_fnames[index])).convert("RGB"))
        mask = self.common_transforms(
            open(join(join(self.path, "masks/"), self.split_fnames[index])).convert("L"))
        mask = (mask > 0.5).float()

        part = mask * image
        masked_image = image - part

        return image, mask, masked_image, part, self.split_fnames[index]


class KvasirSyntheticDataset(Dataset):
    def __init__(self, path, split="train"):
        super(KvasirSyntheticDataset, self).__init__()
        self.path = join(path, "unlabeled-images")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = transforms.Compose([transforms.Resize((400, 400)),
                                                     transforms.ToTensor()
                                                     ])
        self.split = split
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")

    def __len__(self):
        return len(self.split_fnames)

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(join(self.path, "images/"), self.split_fnames[index])).convert("RGB"))
        image = image
        mask = generate_a_mask(imsize=400).T
        mask = torch.tensor(mask > 0.5).float()
        part = mask * image
        masked_image = image - part

        return image, mask, masked_image, part, self.split_fnames[index]


def test_KvasirSegmentationDataset():
    dataset = KvasirSegmentationDataset("Datasets/HyperKvasir", split="test")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        plt.imshow(x.squeeze().T)
        # plt.imshow(y.squeeze().T, alpha=0.5)
        plt.show()

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    print("Segmentation Tests passed")


def test_KvasirClassificationDataset():
    dataset = KvasirClassificationDataset("Datasets/HyperKvasir")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    print("Classification Tests passed")


if __name__ == '__main__':
    test_KvasirSegmentationDataset()
    # test_KvasirClassificationDataset()
