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
from models.ensembles import *
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
        # self.models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
        # self.model_names = ["DeepLab", "FPN", "InductiveNet", "TriUnet", "Unet"]
        self.models = [InductiveNet]
        self.model_names = ["InductiveNet"]

    def parse_experiment_details(self, model_name, eval_method, loss_fn, aug, id, last_epoch=False):
        """
            Note: Supremely messy, since the file structure just sort of evolved
        """
        path = "Predictors"
        if aug != "0":
            path = join(path, "Augmented")
            path = join(path, model_name)
            path = join(path, eval_method)
            if aug == "G":
                path += "inpainter_"
            if loss_fn == "sil":
                path += "consistency"
            else:
                if model_name == "InductiveNet":
                    path += "zaugmentation"  # oops
                else:
                    path += "augmentation"
            path += f"_{id}"


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

    def get_table_data(self, sample_range, id_range, show_reconstruction=False):
        mnv = ModelOfNaturalVariation(0)
        for model_constructor, model_name in zip(self.models, self.model_names):
            for eval_method in [""]:
                for loss_fn in ["j", "sil"]:
                    # for aug in ["0", "V", "G"]:
                    for aug in ["G"]:
                        # for aug in ["0", "V"]:
                        if aug == "G" and loss_fn != "j":
                            continue
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
                            # fig, ax = plt.subplots(ncols=3, nrows=4, sharey=True, figsize=(4, 4), dpi=1000)
                            # fig.subplots_adjust(wspace=0, hspace=0)
                            all_l1s = [[], [], [], []]
                            for dl_idx, dataloader in enumerate(self.dataloaders):
                                print(dl_idx)
                                # if dl_idx != 0:
                                #     continue
                                # seeding ensures SIS metrics are non-stochastic
                                np.random.seed(0)
                                torch.manual_seed(0)
                                random.seed(0)
                                for x, y, _ in dataloader:
                                    img, mask = x.to("cuda"), y.to("cuda")
                                    out = model.predict(img)

                                    with torch.no_grad():
                                        out, reconstruction = model(img)
                                        all_l1s[dl_idx].append(np.mean(np.mean(
                                            np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T))))
                                        #             axis=-1)))
                                        # ax[dl_idx, 0].axis("off")
                                        # ax[dl_idx, 1].axis("off")
                                        # ax[dl_idx, 2].axis("off")
                                        # if dl_idx == 0:
                                        #     ax[dl_idx, 0].title.set_text("Original")
                                        #     ax[dl_idx, 1].title.set_text("Rec.")
                                        #     ax[dl_idx, 2].title.set_text("Difference")
                                        #
                                        # ax[dl_idx, 0].imshow(x[0].cpu().numpy().T)
                                        # ax[dl_idx, 1].imshow(reconstruction[0].cpu().numpy().T)
                                        # ax[dl_idx, 2].imshow(
                                        #     np.mean(np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T),
                                        #             axis=-1))
                                        # all_l1s[dl_idx].append(np.mean(np.mean(np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T),
                                        #             axis=-1)))
                                        # show_reconstruction = False

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
                            fig, ax = plt.subplots(4, 1, sharex=True)
                            colours = ["b", "r", "g", "c"]
                            for index, dataset in enumerate(all_l1s):
                                ax[index].hist(dataset, alpha=0.5, label=self.dataset_names[index],
                                               color=colours[index])
                                ax[index].legend()
                            plt.show()
                            print(f"{full_name} has iou {mean_ious[0, id - 1]} ")
                            # plt.subplots_adjust(wspace=0, hspace=0.01)
                            # plt.show()
                            input()

                            # if mean_ious[0, id - 1] < 0.8:
                            #     print(f"{full_name} has iou {mean_ious[0, id - 1]} ")
                        # with open(f"experiments/Data/pickles/{model_name}_{eval_method}_{loss_fn}_{aug}.pkl",
                        #           "wb") as file:
                        #     pickle.dump({"ious": mean_ious, "sis": sis_matrix}, file)

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


class SingularEnsembleEvaluator:
    def __init__(self, samples=10):
        self.datasets = [
            EtisDataset("Datasets/ETIS-LaribPolypDB"),
            CVC_ClinicDB("Datasets/CVC-ClinicDB"),
            EndoCV2020("Datasets/EndoCV2020"),
        ]
        self.dataloaders = [
                               DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))] + \
                           [DataLoader(dataset) for dataset in self.datasets]
        self.dataset_names = ["HyperKvasir", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
        self.models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
        self.model_names = ["DeepLab", "FPN", "InductiveNet", "TriUnet", "Unet"]
        self.samples = samples

    def get_table_data(self, model_count):
        mnv = ModelOfNaturalVariation(0)
        print(model_count)
        for model_name in self.model_names:
            if model_name != "Unet":
                continue
            mean_ious = np.zeros((len(self.dataloaders), self.samples))
            for i in range(self.samples):
                model = SingularEnsemble(model_name, f"Predictors/Augmented/{model_name}", "consistency", model_count)
                for dl_idx, dataloader in enumerate(self.dataloaders):
                    # seeding ensures SIS metrics are non-stochastic
                    # np.random.seed(0)
                    # torch.manual_seed(0)
                    # random.seed(0)
                    # todo: filter bad predictors
                    for x, y, _ in dataloader:
                        img, mask = x.to("cuda"), y.to("cuda")
                        out = model.predict(img)
                        iou = metrics.iou(out, mask)
                        mean_ious[dl_idx, i] += iou / len(dataloader)
            print(mean_ious)
            with open(f"experiments/Data/pickles/{model_name}-ensemble-{model_count}.pkl",
                      "wb") as file:
                pickle.dump({"ious": mean_ious}, file)


class DiverseEnsembleEvaluator:
    def __init__(self, samples=10):
        self.datasets = [
            EtisDataset("Datasets/ETIS-LaribPolypDB"),
            CVC_ClinicDB("Datasets/CVC-ClinicDB"),
            EndoCV2020("Datasets/EndoCV2020"),
        ]
        self.dataloaders = [
                               DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))] + \
                           [DataLoader(dataset) for dataset in self.datasets]
        self.dataset_names = ["HyperKvasir", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
        self.models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
        self.model_names = ["DeepLab", "FPN", "InductiveNet", "TriUnet", "Unet"]
        self.samples = samples

    def get_table_data(self):
        mnv = ModelOfNaturalVariation(0)
        for type in ["augmentation", "consistency"]:
            mean_ious = np.zeros((len(self.dataloaders), self.samples))
            for i in range(1, self.samples + 1):
                model = DiverseEnsemble(i, "Predictors/Augmented/", type)
                for dl_idx, dataloader in enumerate(self.dataloaders):
                    # seeding ensures SIS metrics are non-stochastic
                    # np.random.seed(0)
                    # torch.manual_seed(0)
                    # random.seed(0)
                    # todo: filter bad predictors
                    for x, y, _ in tqdm(dataloader):
                        img, mask = x.to("cuda"), y.to("cuda")
                        out = model.predict(img)
                        iou = metrics.iou(out, mask)
                        mean_ious[dl_idx, i - 1] += iou / len(dataloader)
                if mean_ious[0, i - 1] < 0.80:
                    print(f"{i} has iou {mean_ious[0, i - 1]}")
                print(mean_ious)
            with open(f"experiments/Data/pickles/diverse-ensemble-{type}.pkl",
                      "wb") as file:
                pickle.dump({"ious": mean_ious}, file)


def write_to_latex_table(pkl_file):
    table_template = open("table_template").read()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    evaluator = ModelEvaluator()
    evaluator.get_table_data(np.arange(0, 6), np.arange(1, 6))
    # evaluator = DiverseEnsembleEvaluator(samples=6)
    # evaluator.get_table_data()
    # evaluator = SingularEnsembleEvaluator()
    # evaluator.get_table_data(5)

    # get_metrics_for_experiment("Augmented", "consistency_1")
