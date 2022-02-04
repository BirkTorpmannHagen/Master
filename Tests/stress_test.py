import random
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataProcessing.etis import EtisDataset
from DataProcessing.hyperkvasir import KvasirSegmentationDataset
from Models.segmentation_models import DeepLab
from Tests.metrics import iou
from utils.logging import log_iou


class AdditiveNoise(nn.Module):
    def __init__(self, noise_factor):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, img):
        return img + torch.randn(img.shape).to("cuda") * random.randint(0, 5) * self.noise_factor

    def __repr__(self):
        return self.__class__.__name__ + '(noise_factor={})'.format(self.noise_factor)


class segmentation_stressors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask):
        return img, mask


def apply_stressors(image, mask):
    img_augs = transforms.Compose([AdditiveNoise(0.05),
                                   transforms.RandomErasing(0.5, scale=(0.02, 0.08)),
                                   transforms.RandomInvert(p=0.1)])
    img_and_mask_augs = segmentation_stressors()
    image = img_augs(image)
    image, mask = img_and_mask_augs(image, mask)
    return image, mask


def stresstesttest():
    dataset = KvasirSegmentationDataset("Datasets/HyperKvasir")
    for x, y, fname in DataLoader(dataset):
        image, mask = apply_stressors(x.to("cuda"), y.to("cuda"))
        plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        # plt.imshow(mask.cpu().numpy().squeeze().T, alpha=0.5)
        plt.show()
        input()


def perform_stresstest(modelpath, stressors=True):
    dataset = EtisDataset("Datasets/ETIS-LaribPolypDB")
    # dataset = KvasirSegmentationDataset("Datasets/HyperKvasir")
    for predictor_name in listdir(modelpath):
        if len(predictor_name.split("-")) == 3:
            model = DeepLab(1).to("cuda")
            model = torch.nn.Sequential(model, torch.nn.Sigmoid())

            model.eval()
            test = torch.load(join(modelpath, predictor_name))
            # print(test)
            model.load_state_dict(test)
            ious = torch.empty((0,))

            for x, y, fname in tqdm(DataLoader(dataset)):
                if stressors:
                    image, mask = apply_stressors(x.to("cuda"), y.to("cuda"))
                else:
                    image, mask = x.to("cuda"), y.to("cuda")
                with torch.no_grad():
                    output = model(image)
                    batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                    ious = torch.cat((ious, batch_ious.flatten()))
            log_iou("logs/kvasir-no-pretrain-baseline.log", -1, predictor_name.split("-")[-1], ious)


if __name__ == '__main__':
    perform_stresstest("Predictors/BaselineDeepLab", stressors=False)
    # EtisDataset("")
