from os import listdir
from os.path import join
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.logging import log_full
from evaluation.metrics import SegmentationInconsistencyScore
from data.etis import EtisDataset
from data.hyperkvasir import KvasirSegmentationDataset, KvasirMNVset
from data.endocv import EndoCV2020
from data.cvc import CVC_ClinicDB
from models.segmentation_models import *
from evaluation.metrics import iou
from torch.utils.data import DataLoader
from losses.consistency_losses import NakedConsistencyLoss
from perturbation.model import ModelOfNaturalVariation


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
        self.dataset_names = ["HyperKvasir", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]

    def get_consistency_auc(self, model, t_range, model_name):
        mnv = ModelOfNaturalVariation(0).to("cuda")
        consistency = SegmentationInconsistencyScore()
        cons_matrix = np.zeros((len(self.dataloaders), len(t_range)))
        for dl_idx, dataloader in enumerate(self.dataloaders):
            for tmp_idx, temp in enumerate(t_range):
                consistencies = []
                mnv.set_temp(temp)
                for x, y, _ in dataloader:
                    img, mask = x.to("cuda"), y.to("cuda")
                    mnv.temp = temp
                    aug_img, aug_mask = mnv(img, mask)
                    out = model.predict(img)
                    aug_out = model.predict(aug_img)
                    consistencies.append(consistency(mask, aug_mask, out, aug_out).item())
                cons_matrix[dl_idx, tmp_idx] = np.mean(consistencies)
            plt.plot(t_range, cons_matrix[dl_idx], label=self.dataset_names[dl_idx])
            print(cons_matrix[dl_idx])
        plt.legend()
        plt.title(model_name)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.show()
        return cons_matrix, t_range

    def get_ious(self, model):
        all_ious = np.zeros(len(self.dataloaders))
        print(all_ious.shape)
        for idx, dataset in enumerate(self.dataloaders):
            dataset_ious = torch.zeros((len(self.dataloaders),))
            for x, y, fname in dataset:
                image = x.to("cuda")
                mask = y.to("cuda")
                output = model.predict(image)
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                dataset_ious = torch.cat((dataset_ious, batch_ious.flatten()))
            all_ious[idx] = np.mean(dataset_ious.cpu().numpy())
            # print(np.mean(dataset_ious.cpu().numpy()))
        return all_ious


def get_metrics(type, experiment):
    models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
    evaluator = ModelEvaluator()
    model_wise_results = {}
    for model in models:
        path = join("Predictors", type, model.__name__)
        auc_dict = {}
        iou_dict = {}
        for pred_fname in tqdm([i for i in listdir(path) if experiment in i]):
            print(f"Evaluating {path}/{pred_fname}")
            predictor = model().to("cuda")
            predictor.load_state_dict(torch.load(join(path, pred_fname), map_location=torch.device('cpu')))
            ious = evaluator.get_ious(predictor)
            consistency_curves, bins = evaluator.get_consistency_auc(predictor, np.linspace(0, 1, 10), pred_fname)
            consistency_aucs = np.sum(consistency_curves, axis=1) / len(bins)
            print(consistency_aucs)
            auc_dict[pred_fname] = consistency_aucs
            iou_dict[pred_fname] = ious
        model_wise_results[model.__name__] = [iou_dict, auc_dict]
    with open("results.pkl", "wb") as file:
        pkl.dump(model_wise_results, file)
    return model_wise_results


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    get_metrics("Augmented", "consistency_")
