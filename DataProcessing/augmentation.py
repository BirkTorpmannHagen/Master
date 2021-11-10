import torch.nn
import torchvision.transforms.functional as F
from torchvision import transforms
from random import randint
import albumentations as A


def pipeline_tranforms():
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])


def albumentation_pixelwise_transforms():
    return A.Compose([
        A.ImageCompression(1, 50, p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.01, p=0.1),
        A.RandomSnow(p=0.1),
        A.GlassBlur(p=0.1),
        A.CLAHE(p=0.1)

    ])


def albumentation_mask_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.25),
        A.ElasticTransform(p=0.1),
        A.OpticalDistortion(p=0.1)
    ])


def image_transforms():
    return transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAdjustSharpness(0.9, 0.1),
        transforms.RandomAutocontrast()
    ])


class Quantize(torch.nn.Module):
    # TODO this transform is broken for some reason.
    def __init__(self, p, gray_level_min, gray_level_max):
        super(Quantize, self).__init__()
        self.p = p
        self.gmin = gray_level_min
        self.gmax = gray_level_max

    def forward(self, x):
        new_batch = torch.zeros_like(x)
        if torch.rand(1) < self.p:
            for batch_idx in range(x.shape[0]):
                img = x[batch_idx].squeeze()
                img = transforms.ToPILImage()(img)
                img = img.quantize(randint(self.gmin, self.gmax))
                img = transforms.ToTensor()(img)
                new_batch[batch_idx] = img
        return new_batch


class SegAugmentations(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.
            mask PIL Image or Tensor): Image to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
            :param img:
            :param mask:
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask)
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(mask)

        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


if __name__ == '__main__':
    pass
