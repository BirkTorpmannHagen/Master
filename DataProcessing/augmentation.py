import torch.nn
import torchvision.transforms.functional as F
from torchvision import transforms


def pipeline_tranforms():
    return transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])


def image_transforms():
    return transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAdjustSharpness(0.9, 0.1),
        transforms.RandomAutocontrast()
    ])


class seg_augmentations(torch.nn.Module):
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
