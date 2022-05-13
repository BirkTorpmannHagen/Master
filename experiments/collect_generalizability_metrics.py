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
        self.dataset_names = ["Kvasir-Seg", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
        # self.models = [DeepLab, FPN, InductiveNet, TriUnet, Unet]
        # self.model_names = ["DeepLab", "FPN", "InductiveNet", "TriUnet", "Unet"]
        self.models = [FPN]
        self.model_names = ["FPN"]
        # self.models = [InductiveNet]
        # self.model_names = ["InductiveNet"]

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
                if model_name == "InductiveNet" and aug != "V":
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

    def get_table_data(self, sample_range, id_range, show_reconstruction=False, show_consistency_examples=False):
        mnv = ModelOfNaturalVariation(1)
        for model_constructor, model_name in zip(self.models, self.model_names):
            for eval_method in [""]:
                for loss_fn in ["j", "sil"]:
                    for aug in ["0", "V", "G"]:
                        sis_matrix = np.zeros((len(self.dataloaders), len(id_range)))
                        mean_ious = np.zeros((len(self.dataloaders), len(id_range)))
                        for id in id_range:
                            try:
                                state_dict, full_name = self.parse_experiment_details(model_name, eval_method, loss_fn,
                                                                                      aug,
                                                                                      id)
                                model = model_constructor().to("cuda")
                                model.load_state_dict(state_dict)
                                print(f"Evaluating {full_name}")
                            except FileNotFoundError:
                                print(f"{model_name}-{eval_method}-{loss_fn}-{aug}-{id} not found, continuing...")
                                continue
                            # fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(4, 3), dpi=1000)
                            # fig.subplots_adjust(wspace=0, hspace=0)
                            for dl_idx, dataloader in enumerate(self.dataloaders):
                                # print("dl idx: ", dl_idx)
                                # seeding ensures SIS metrics are non-stochastic
                                np.random.seed(0)
                                torch.manual_seed(0)
                                random.seed(0)

                                for i, (x, y, _) in enumerate(dataloader):
                                    img, mask = x.to("cuda"), y.to("cuda")
                                    aug_img, aug_mask = mnv(img, mask)
                                    out = model.predict(img)
                                    aug_out = model.predict(aug_img)

                                    if dl_idx == 0 and show_consistency_examples:
                                        fig, ax = plt.subplots(2, 3)
                                        xor = lambda a, b: a * (1 - b) + b * (1 - a)
                                        diff = xor(xor(out, aug_out), xor(mask, aug_mask))
                                        union = torch.clamp((out + aug_out + mask + aug_mask), 0, 1)
                                        fig.suptitle(
                                            f"Inconsistency: {metrics.sis(aug_mask, mask, aug_out, out)}")
                                        ax[0, 0].imshow(img[0].cpu().numpy().T)
                                        ax[0, 0].set_title("Unperturbed Image")
                                        ax[1, 0].imshow(aug_img[0].cpu().numpy().T)
                                        ax[1, 0].set_title("Perturbed Image")
                                        ax[0, 1].imshow(out[0].cpu().numpy().T)
                                        ax[0, 1].set_title("Unperturbed Output")

                                        ax[1, 1].imshow(aug_out[0].cpu().numpy().T)
                                        ax[1, 1].set_title("Perturbed Output")
                                        print(ax[1, 1].get_position())
                                        ax[0, 2].imshow(diff[0].cpu().numpy().T, cmap="viridis")
                                        ax[0, 2].set_title("Inconsistency")
                                        # print(ax[0, 2].get_position())
                                        ax[0, 2].set_position([0.67, 0.34, 0.90, 0])

                                        # ax[1, 2].imshow(intersection[0].cpu().numpy().T)
                                        # ax[1, 2].set_title("Consistency")

                                        for axi in ax.flatten():
                                            axi.set_yticks([])
                                            axi.set_xticks([])
                                            axi.spines['top'].set_visible(False)
                                            axi.spines['right'].set_visible(False)
                                            axi.spines['bottom'].set_visible(False)
                                            axi.spines['left'].set_visible(False)
                                        plt.subplots_adjust(wspace=0.1, hspace=0.1)
                                        plt.show()

                                    if i == 0 and dl_idx == 0 and show_consistency_examples:
                                        with torch.no_grad():
                                            fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(2, 2), dpi=1000,
                                                                   sharex=True, sharey=True)
                                            out, reconstruction = model(img)
                                            img_n = img + torch.rand_like(img) / 2.5
                                            out_n, reconstruction_n = model(img_n)
                                            xor = lambda a, b: a * (1 - b) + b * (1 - a)
                                            diff = xor(xor(out, aug_out), xor(mask, aug_mask))

                                            union = torch.clamp((out + out_n), 0, 1)
                                            ax[0, 0].imshow(img[0].cpu().numpy().T)
                                            ax[0, 0].set_title("Unperturbed Image")
                                            ax[1, 0].imshow(img_n[0].cpu().numpy().T)
                                            ax[1, 0].set_title("Perturbed Image")
                                            ax[0, 1].imshow(out[0].cpu().numpy().T)
                                            ax[0, 1].set_title("Unperturbed Output")

                                            ax[1, 1].imshow(out_n[0].cpu().numpy().T)
                                            ax[1, 1].set_title("Perturbed Output")

                                            ax[0, 2].imshow(diff[0].cpu().numpy().T)
                                            ax[0, 2].set_title("Inconsistency")

                                            # ax[1, 2].imshow(intersection[0].cpu().numpy().T)
                                            # ax[1, 2].set_title("Consistency")

                                            for axi in ax.flatten():
                                                axi.title.set_size(3.5)

                                                axi.set_yticks([])
                                                axi.set_xticks([])
                                                axi.spines['top'].set_visible(False)
                                                axi.spines['right'].set_visible(False)
                                                axi.spines['bottom'].set_visible(False)
                                                axi.spines['left'].set_visible(False)
                                            plt.subplots_adjust(wspace=0.1, hspace=0.1)
                                            plt.savefig("consistency_examples.png")
                                            plt.show()

                                            # print(torch.sum(diff) / torch.sum(union))
                                            # print(torch.sum(intersection) / torch.sum(union))

                                            input()
                                    if show_reconstruction and i == 0:

                                        with torch.no_grad():
                                            out, reconstruction = model(img)
                                            # all_l1s[dl_idx].append(np.mean(np.mean(
                                            #     np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T))))
                                            #             axis=-1)))
                                            # ax[0, dl_idx].axis("off")
                                            # ax[1, dl_idx].axis("off")
                                            # ax[2, dl_idx].axis("off")
                                            # ax[3, dl_idx].axis("off")

                                            # ax[0, dl_idx].set_xlabel(self.dataset_names[dl_idx])
                                            for i in range(4):
                                                ax[0, i].title.set_text(self.dataset_names[i])
                                                ax[0, i].title.set_size(8)
                                            ax[0, 0].set_ylabel("Original", fontsize=8)
                                            ax[1, 0].set_ylabel("Reconstruction", fontsize=8)
                                            ax[2, 0].set_ylabel("L1", fontsize=8)
                                            for axi in ax.flatten():
                                                axi.set_yticks([])
                                                axi.set_xticks([])
                                                axi.spines['top'].set_visible(False)
                                                axi.spines['right'].set_visible(False)
                                                axi.spines['bottom'].set_visible(False)
                                                axi.spines['left'].set_visible(False)

                                            ax[0, dl_idx].imshow(x[0].cpu().numpy().T)
                                            ax[1, dl_idx].imshow(reconstruction[0].cpu().numpy().T)
                                            ax[2, dl_idx].imshow(
                                                np.mean(
                                                    np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T),
                                                    axis=-1))
                                            # all_l1s[dl_idx].append(np.mean(
                                            #     np.mean(np.abs(reconstruction[0].cpu().numpy().T - x[0].cpu().numpy().T),
                                            #             axis=-1)))

                                    iou = metrics.iou(out, mask)
                                    # consistency
                                    sis = metrics.sis(aug_mask, mask, aug_out, out)
                                    sis_matrix[dl_idx, id - id_range[0]] += sis / len(dataloader)
                                    mean_ious[dl_idx, id - id_range[0]] += iou / len(dataloader)

                            # print(
                            #     f"{full_name} has iou {mean_ious[0, id - 1]} and consistency {sis_matrix[0, id - 1]} ")

                            if mean_ious[0, id - 1] < 0.8:
                                print(f"{full_name} has iou {mean_ious[0, id - 1]} ")
                        with open(f"experiments/Data/pickles/{model_name}_{eval_method}_{loss_fn}_{aug}.pkl",
                                  "wb") as file:
                            pickle.dump({"ious": mean_ious, "sis": sis_matrix}, file)


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
        for type in ["vanilla"]:
            for model_name in self.model_names:
                # if model_name != "TriUnet":
                #     continue
                print(model_name)
                mean_ious = np.zeros((len(self.dataloaders), self.samples))
                constituents = {}
                for i in range(self.samples):
                    model = SingularEnsemble(model_name, type, model_count)
                    constituents[i] = model.get_constituents()
                    for dl_idx, dataloader in enumerate(self.dataloaders):
                        for x, y, _ in tqdm(dataloader):
                            img, mask = x.to("cuda"), y.to("cuda")
                            out = model.predict(img, threshold=True)

                            iou = metrics.iou(out, mask)
                            mean_ious[dl_idx, i] += iou / len(dataloader)
                    del model  # avoid memory issues
                print(mean_ious)
                if type == "consistency":
                    with open(f"experiments/Data/pickles/{model_name}-ensemble-{model_count}.pkl",
                              "wb") as file:
                        pickle.dump({"ious": mean_ious, "constituents": constituents}, file)
                else:
                    with open(f"experiments/Data/pickles/{model_name}-ensemble-{model_count}-{type}.pkl",
                              "wb") as file:
                        pickle.dump({"ious": mean_ious, "constituents": constituents}, file)


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
        for type in ["vanilla"]:
            mean_ious = np.zeros((len(self.dataloaders), self.samples))
            constituents = {}
            for i in range(1, self.samples + 1):
                model = DiverseEnsemble(i, type)
                constituents[i] = model.get_constituents()
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
                pickle.dump({"ious": mean_ious, "constituents": constituents}, file)


def write_to_latex_table(pkl_file):
    table_template = open("table_template").read()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    # evaluator = ModelEvaluator()
    # evaluator.get_table_data(np.arange(0, 10), np.arange(1, 11), show_reconstruction=False,
    #                          show_consistency_examples=False)
    evaluator = DiverseEnsembleEvaluator(samples=10)
    evaluator.get_table_data()
    # evaluator = SingularEnsembleEvaluator()
    # evaluator.get_table_data(5)

    # get_metrics_for_experiment("Augmented", "consistency_1")
