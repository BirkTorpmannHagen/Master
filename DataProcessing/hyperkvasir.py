from os import listdir
from os.path import join

import torch.utils.data
from PIL.Image import open
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms

from DataProcessing.augmentation import vanilla_augmentations


class KvasirClassificationDataset(Dataset):
    """
    Dataset class that fetches images with the associated pathological class labels for use in Pretraining
    """

    def __init__(self, path):
        super(KvasirClassificationDataset, self).__init__()
        self.path = join(path, "labeled-images/lower-gi-tract/pathological-findings")
        self.transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor()
        ])

        self.label_names = listdir(self.path)
        self.num_classes = len(self.label_names)
        self.fname_class_dict = {}
        i = 0
        for i, label in enumerate(self.label_names):
            class_path = join(self.path, label)
            for fname in listdir(class_path):
                self.fname_class_dict[fname] = label
        self.index_dict = dict(zip(self.label_names, range(self.num_classes)))

    def __len__(self):
        return len(self.fname_class_dict)

    def __getitem__(self, item):
        fname, label = list(self.fname_class_dict.items())[item]
        onehot = one_hot(torch.tensor(self.index_dict[label]), num_classes=self.num_classes)
        image = open(join(join(self.path, label), fname)).convert("RGB")
        image = self.transforms(image)
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
        self.train_transforms = transforms.Compose([])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        image = (open(join(join(self.path, "images/"), self.fnames[index])).convert("RGB"))
        mask = (open(join(join(self.path, "masks/"), self.fnames[index])).convert("RGB"))
        vanilla_aug = vanilla_augmentations()
        image, mask = vanilla_aug(image, mask)
        mask = self.common_transforms(mask)[0].unsqueeze(0)
        image = self.common_transforms(image)
        image = self.train_transforms(image)
        mask = (mask > 0.5).float()
        return image, mask, self.fnames[index]


def test_KvasirSegmentationDataset():
    dataset = KvasirSegmentationDataset("Data")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        print(y)
    print("Classification Tests passed")


def test_KvasirClassificationDataset():
    dataset = KvasirClassificationDataset("Data")
    for x, y, fname in torch.utils.data.DataLoader(dataset):
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        print(y)
    print("Segmentation Tests passed")


if __name__ == '__main__':
    # test_KvasirSegmentationDataset()
    test_KvasirClassificationDataset()
