from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL.Image import open
from torch.utils.data import Dataset
import DataProcessing.augmentation as aug


class EndoCV2020(Dataset):
    def __init__(self, root_directory):
        super(EndoCV2020, self).__init__()
        self.root = root_directory
        self.mask_fnames = listdir(join(self.root, "masksPerClass", "polyp"))
        self.mask_locs = [join(self.root, "masksPerClass", "polyp", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "originalImages", i.replace("_polyp", "").replace(".tif", ".jpg")) for i in
                         self.mask_fnames]
        self.common_transforms = aug.pipeline_tranforms()

    def __getitem__(self, idx):
        mask = self.common_transforms(open(self.mask_locs[idx]))
        image = self.common_transforms(open(self.img_locs[idx]))
        return image, mask, self.mask_fnames[idx]

    def __len__(self):
        return len(self.mask_fnames)


if __name__ == '__main__':
    dataset = EndoCV2020("Datasets/EndoCV2020-Endoscopy-Disease-Detection-Segmentation-subChallenge_data")
    for img, mask, fname in dataset:
        plt.imshow(img.T)
        plt.imshow(mask.T, alpha=0.5)
        plt.show()
        input()
    print("done")
