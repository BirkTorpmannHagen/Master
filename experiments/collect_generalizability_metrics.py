import pickle
from os import listdir
from os.path import join
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from data.etis import EtisDataset
from data.hyperkvasir import KvasirSegmentationDataset
from data.endocv import EndoCV2020
from data.cvc import CVC_ClinicDB
from models.segmentation_models import *
from evaluation import metrics
from torch.utils.data import DataLoader
from perturbation.model import ModelOfNaturalVariation
import random


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
        self.models = [FPN, InductiveNet, TriUnet, Unet]
        self.model_names = ["FPN", "InductiveNet", "TriUnet", "Unet"]
        # self.models = [DeepLab]
        # self.model_names = ["DeepLab"]

    def collect_stats(self, model, predictor_name, sample_range):
        mnv = ModelOfNaturalVariation(0)
        sis_matrix = np.zeros((len(self.dataloaders), len(sample_range)))
        ap_matrix = np.zeros((len(self.dataloaders), 2, len(sample_range)))
        ious = np.zeros(len(self.dataloaders))

        for dl_idx, dataloader in enumerate(self.dataloaders):
            for x, y, _ in dataloader:
                img, mask = x.to("cuda"), y.to("cuda")
                out = model.predict(img)

                # sis_auc metric
                for idx, temp in enumerate(sample_range):
                    mnv.set_temp(temp)
                    aug_img, aug_mask = mnv(img, mask)
                    aug_out = model.predict(aug_img)
                    sis_matrix[dl_idx, idx] += np.mean(metrics.sis(mask, out, aug_mask, aug_out).item()) / len(
                        sample_range)  # running mean

                # PR-curve
                for idx, thresh in enumerate(sample_range):
                    precision = metrics.precision(out, mask, thresh)
                    recall = metrics.recall(out, mask, thresh)
                    ap_matrix[dl_idx, 0, idx] += precision / len(sample_range)
                    ap_matrix[dl_idx, 1, idx] += recall / len(sample_range)
        with open(f"{model.__name__}_{predictor_name}_results.pkl", "wb") as file:
            pkl.dump({"sis_matrix": sis_matrix, "ap_matrix": ap_matrix, "ious": ious}, file)
        return {"sis_matrix": sis_matrix, "ap_matrix": ap_matrix, "ious": ious}

    def parse_experiment_details(self, model_name, eval_method, loss_fn, aug, id, last_epoch=False):
        """
            Note: Supremely messy, since the file structure just sort of evolved
        """
        path = "Predictors"
        if aug != "0":
            path = join(path, "Augmented")
            path = join(path, model_name)
            path = join(path, eval_method)
            if loss_fn == "sil":
                path += "consistency"
            else:
                path += "augmentation"
            path += f"_{id}"
            if aug == "G":
                path += "_inpainter"

        else:
            path = join(path, "Vanilla")
            path = join(path, model_name)
            path = join(path, "vanilla")
            path += f"_{id}"
            if eval_method == "maximum_consistency":
                path += "-maximum-consistency"
            elif last_epoch:
                path += "_last_epoch"

        return torch.load(path), path

    def get_table_data(self, sample_range, id_range):
        mnv = ModelOfNaturalVariation(0)
        for model_constructor, model_name in zip(self.models, self.model_names):
            for eval_method in [""]:
                for loss_fn in ["j", "sil"]:
                    for aug in ["0", "V", "G"]:
                        sis_matrix = np.zeros((len(self.dataloaders), len(id_range), len(sample_range)))
                        mean_ious = np.zeros((len(self.dataloaders), len(id_range)))
                        if aug == "0" and loss_fn == "sil":
                            continue
                        for id in id_range:
                            # print(eval_method, loss_fn, aug, id)
                            try:
                                state_dict, full_name = self.parse_experiment_details(model_name, eval_method, loss_fn,
                                                                                      aug,
                                                                                      id)
                                model = model_constructor().to("cuda")
                                model.load_state_dict(state_dict)
                            except FileNotFoundError:
                                print(f"{model_name}-{eval_method}-{loss_fn}-{aug}-{id} not found, continuing...")
                                continue
                            for dl_idx, dataloader in enumerate(self.dataloaders):
                                # seeding ensures SIS metrics are non-stochastic
                                np.random.seed(0)
                                torch.manual_seed(0)
                                random.seed(0)
                                for x, y, _ in tqdm(dataloader):
                                    img, mask = x.to("cuda"), y.to("cuda")
                                    out = model.predict(img)
                                    iou = metrics.iou(out, mask)
                                    # dataset_ious[sample_idx] = iou
                                    mean_ious[dl_idx, id - id_range[0]] += iou / len(dataloader)
                                    # sis_auc metric
                                    # for idx, temp in enumerate(sample_range):
                                    #     mnv.set_temp(temp)
                                    #     aug_img, aug_mask = mnv(img, mask)
                                    #     # if temp == 1:
                                    #     #     plt.imshow(aug_img[0].cpu().numpy().T)
                                    #     #     plt.show()
                                    #     aug_out = model.predict(aug_img)
                                    #     sis_matrix[dl_idx, id - id_range[0], idx] += np.mean(
                                    #         metrics.sis(mask, out, aug_mask, aug_out).item()) / len(
                                    #         dataloader)  # running mean
                                    # break
                        with open(f"experiments/Data/pickles/{model_name}_{eval_method}_{loss_fn}_{aug}.pkl",
                                  "wb") as file:
                            pickle.dump({"ious": mean_ious, "sis": sis_matrix}, file)
                        # torch.save(dataset_ious,
                        #                    f"experiments/Data/pickles/ious_{self.dataset_names[dl_idx]}_"
                        #                    f"{model_name}_{eval_method}_{loss_fn}_{aug}")
                        # torch.save(sis_matrix,
                        #                f"experiments/Data/pickles/sis-scores_{model_name}_{eval_method}_{loss_fn}_{aug}")

                        # print(f"IOUS for {full_name}: {mean_ious}")
                        # for i, name in enumerate(self.dataset_names):
                        #     plt.plot(sample_range, sis_matrix[i], label=name)
                        # plt.title(f"SIS-curve for {full_name}")
                        # plt.legend()
                        # plt.xlabel("SIS")
                        # plt.xlabel("Augmentation Severity")
                        # plt.ylim((0, 1))
                        # plt.xlim((0, 1))
                        # plt.show()


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
            stats = evaluator.collect_stats(predictor, pred_fname, np.linspace(0.1, 0.9, 9))
            model_wise_results[model.__name__].append(stats)
    with open(f"{experiment}-results.pkl", "wb") as file:
        pkl.dump(model_wise_results, file)
    return model_wise_results


def write_to_latex_table(pkl_file):
    table_template = open("table_template").read()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    write_to_latex_table(0)
    evaluator = ModelEvaluator()
    evaluator.get_table_data(np.linspace(0, 1, 11), range(1, 11))

    # get_metrics_for_experiment("Augmented", "consistency_1")
