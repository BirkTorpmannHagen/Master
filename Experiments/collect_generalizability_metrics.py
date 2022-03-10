from os import listdir
from os.path import join
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.logging import log_full
from DataProcessing.etis import EtisDataset
from DataProcessing.hyperkvasir import KvasirSegmentationDataset, KvasirMNVset
from DataProcessing.endocv import EndoCV2020
from DataProcessing.cvc import CVC_ClinicDB
from models.segmentation_models import *
from evaluation.metrics import iou
from torch.utils.data import DataLoader
from losses.consistency_losses import NakedConsistencyLoss
from model_of_natural_variation.model import ModelOfNaturalVariation


class ModelEvaluator:
    def __init__(self):
        self.datasets = [
            EtisDataset("Datasets/ETIS-LaribPolypDB"),
            CVC_ClinicDB("Datasets/CVC-ClinicDB"),
            EndoCV2020("Datasets/EndoCV2020"),
        ]
        self.dataloaders = [
                               DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))] + \
                           [DataLoader(dataset) for dataset in self.datasets]

    def get_consistency_auc(self, model, t_range):
        mnv = ModelOfNaturalVariation(0)
        consistency = NakedConsistencyLoss()
        cons_matrix = np.array((len(self.datasets), len(t_range)))
        for dl_idx, dataloader in enumerate(self.dataloaders):
            for tmp_idx, temp in enumerate(t_range):
                consistencies = []
                for x, y, _ in dataloader:
                    img, mask = x, y
                    mnv.temp = temp
                    aug_img, aug_mask = mnv(img, mask)
                    out = model.predict(img)
                    aug_out = model.predict(aug_img)
                    consistencies.append(consistency(mask, out, aug_mask, aug_out).item())
                cons_matrix[dl_idx, tmp_idx] = np.mean(consistencies)
        return cons_matrix

    def get_ious(self, model):
        all_ious = np.zeros(len(self.dataloaders))
        print(all_ious.shape)
        for idx, dataset in enumerate(self.dataloaders):
            dataset_ious = torch.empty((len(self.dataloaders),))
            for x, y, fname in tqdm(dataset):
                image = x
                mask = y
                output = model.predict(image)
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                dataset_ious = torch.cat((dataset_ious, batch_ious.flatten()))
            all_ious[idx] = np.mean(dataset_ious.cpu().numpy())
        return all_ious


def get_metrics(type, experiment):
    models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
    evaluator = ModelEvaluator()
    for model in models:
        path = join("Predictors", type, model.__name__)
        for pred_fname in [i for i in listdir(path) if experiment in i]:
            predictor = model()
            predictor.load_state_dict(torch.load(join(path, pred_fname), map_location=torch.device('cpu')))
            ious = evaluator.get_ious(predictor)
            print(ious)
            consistency_auc = evaluator.get_consistency_auc(predictor, np.linspace(0, 1, 10))
            print(consistency_auc)


if __name__ == '__main__':
    get_metrics("Augmented", "consistency")
