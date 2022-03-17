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
from data.etis import EtisDataset
from data.hyperkvasir import KvasirSegmentationDataset, KvasirMNVset
from data.endocv import EndoCV2020
from data.cvc import CVC_ClinicDB
from models.segmentation_models import *
from evaluation import metrics
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
                    consistencies.append(metrics.sis(mask, aug_mask, out, aug_out).item())
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
                batch_ious = torch.Tensor([metrics.iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                dataset_ious = torch.cat((dataset_ious, batch_ious.flatten()))
            all_ious[idx] = np.mean(dataset_ious.cpu().numpy())
            # print(np.mean(dataset_ious.cpu().numpy()))
        return all_ious

    def get_pr_curve_stats(self):
        for idx, dataset in enumerate(self.dataloaders):
            for x, y, fname in tqdm(dataset):
                pass

    def collect_stats(self, model, predictor_name, sample_range):
        mnv = ModelOfNaturalVariation(0)
        sis_matrix = np.array((len(self.datasets), len(sample_range)))
        ap_matrix = np.array((len(self.datasets),2, len(sample_range)))
        ious = np.zeros(len(self.dataloaders))

        for dl_idx, dataloader in enumerate(self.dataloaders):
            for x, y, _ in dataloader:
                img, mask = x.to("cuda"), y.to("cuda")
                out = model.predict(img)

                #sis_auc metric
                for idx, temp in enumerate(sample_range):
                    mnv.set_temp(temp)
                    aug_img, aug_mask = mnv(img, mask)
                    aug_out = model.predict(aug_img)
                    sis_matrix[dl_idx, temp] += np.mean(metrics.sis(mask, out, aug_mask, aug_out).item())/len(sample_range) #running mean

                #PR-curve
                for idx, thresh in enumerate(sample_range):
                    precision = metrics.precision(out, mask, thresh)
                    recall = metrics.recall(out, mask, thresh)
                    ap_matrix[dl_idx, 0, idx]+=precision/len(sample_range)
                    ap_matrix[dl_idx, 1, idx]+=recall/len(sample_range)
        with open(f"{model.__name__}_{predictor_name}_results.pkl", "wb") as file:
            pkl.dump({"sis_matrix":sis_matrix, "ap_matrix":ap_matrix, "ious":ious}, file)
        return {"sis_matrix":sis_matrix, "ap_matrix":ap_matrix, "ious":ious}


def get_metrics_for_experiment(type, experiment):
    models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
    evaluator = ModelEvaluator()
    model_wise_results = dict(zip([i.__name__ for i in models], [[] for _ in models]))
    for model in models:
        path = join("Predictors", type, model.__name__)
        predictor = model().to("cuda")
        for pred_fname in tqdm([i for i in listdir(path) if experiment in i]):
            print(f"Evaluating {path}/{pred_fname}")
            predictor.load_state_dict(torch.load(join(path, pred_fname)))
            stats=evaluator.collect_stats(predictor, pred_fname, np.linspace(0,1,11))
            model_wise_results[model.__name__].append(stats)
    with open(f"{experiment}-results.pkl", "wb") as file:
        pkl.dump(model_wise_results, file)
    return model_wise_results


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    get_metrics_for_experiment("Augmented", "consistency_")
