from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL.Image import open
from torch.utils.data import Dataset
import data.augmentation as aug

from PIL import Image


class CVC_ClinicDB(Dataset):
    def __init__(self, root_directory):
        super(CVC_ClinicDB, self).__init__()
        self.root = root_directory
        self.mask_fnames = listdir(join(self.root, "Ground Truth"))
        self.mask_locs = [join(self.root, "Ground Truth", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "Original", i) for i in
                         self.mask_fnames]
        self.common_transforms = aug.pipeline_tranforms()

    def __getitem__(self, idx):
        mask = self.common_transforms(open(self.mask_locs[idx]))
        image = self.common_transforms(open(self.img_locs[idx]))
        return image, mask, self.mask_fnames[idx]

    def __len__(self):
        return len(self.mask_fnames)


if __name__ == '__main__':
    dataset = CVC_ClinicDB("Datasets/CVC-ClinicDB")
    for img, mask, fname in dataset:
        plt.imshow(img.T)
        plt.imshow(mask.T, alpha=0.5)
        plt.show()
        input()
    print("done")
