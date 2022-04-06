import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from utils.formatting import SafeDict
from scipy.stats import wasserstein_distance
from models.segmentation_models import *


def training_plot(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("Training Plot Sample")
    plt.xlabel("Epochs")
    plt.ylabel("Jaccard Loss")
    plt.xlim((0, 300))
    plt.ylim((0, 1))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Training Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss")
    # plt.plot(log_df["epoch"], log_df["ood_iou"], label="Etis-LaribDB iou")
    plt.legend()
    plt.show()


def ood_correlations(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("SIS-OOD correlation")
    plt.xlabel("SIS")
    plt.ylabel("Etis-LaribDB OOD performance")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.scatter(log_df["consistency"], log_df["ood_iou"], label="Consistency")
    plt.scatter(log_df["iid_test_iou"], log_df["ood_iou"], label="IID IoU")

    plt.legend()
    plt.show()


def ood_v_epoch(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("Training Plot Sample")
    plt.xlabel("Epochs")
    plt.ylabel("SIL")
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.plot(log_df["epoch"], log_df["consistency"], label="consistency")
    plt.plot(log_df["epoch"], log_df["ood_iou"], label="ood iou")
    plt.legend()
    plt.show()


def modelwise_boxplot():
    results = torch.load("experiments/Data/pickles/ious_CVC-ClinicDB_DeepLab__j_0")  # vanilla


def get_boxplots_for_models():
    """
    box plot for comparing model performance. Considers d% reduced along datasets, split according to experiments
    and models
    :return:
    """
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN, Unet, InductiveNet, TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "0" in fname:
            with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
                model = fname.split("_")[0]
                if model == "InductiveNet":
                    model = "DD-DeepLabV3+"
                data = pickle.load(file)
                datasets, samples = data["ious"].shape
                kvasir_ious = data["ious"][0]
                # print(kvasir_ious)
                mean_iid_iou = np.median(kvasir_ious)
                print(mean_iid_iou)
                if "maximum_consistency" in fname:
                    continue
                for i in range(datasets):
                    # if i==0:
                    #     continue
                    for j in range(samples):
                        if data["ious"][i, j] < 0.25 or data["ious"][0][j] < 0.75:
                            print(fname, "-", 2 + j)
                            continue
                        dataset.append([dataset_names[i], model, data["ious"][i, j]])

                        # dataset.append([dataset_names[i], model, 100*(data["ious"][i, j]-mean_iid_iou)/mean_iid_iou])

    dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "\u0394%IoU"])
    sns.barplot(x="Dataset", y="\u0394%IoU", hue="Model", data=dataset)
    plt.ylim((0, 1))
    plt.show()


def get_variances_for_models():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN, Unet, InductiveNet, TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "maximum_consistency" in fname:
            continue
        if "0" in fname:
            with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
                model = fname.split("_")[0]
                if model == "InductiveNet":
                    model = "DD-DeepLabV3+"
                data = pickle.load(file)
                datasets, samples = data["ious"].shape

                if "maximum_consistency" in fname:
                    continue
                for i in range(datasets):
                    # if i == 0:
                    #     continue

                    for j in range(samples):
                        if data["ious"][0][j] < 0.75:
                            print(fname, "-", j)
                            continue
                        if i == 3 and model == "InductiveNet":
                            print("inductivenet", data["ious"][i, j])
                        if i == 3 and model == "DeepLab":
                            print("DeepLab", data["ious"][i, j])

                        dataset.append([dataset_names[i], model, data["ious"][i, j]])

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "Coefficient of Std.Dev"])
    std_dataset = iou_dataset.groupby(["Model", "Dataset"]).std() / iou_dataset.groupby(["Model", "Dataset"]).mean()
    std_dataset = std_dataset.reset_index()
    print(std_dataset)
    plt.ylim((0, 0.15))
    sns.barplot(x="Dataset", y="Coefficient of Std.Dev", hue="Model", data=std_dataset)
    plt.show()


def plot_parameters_sizes():
    models = [DeepLab, FPN, InductiveNet, Unet, TriUnet]
    model_names = ["DeepLab", "FPN", "InductiveNet", "Unet", "TriUnet"]
    for model_name, model_c in zip(model_names, models):
        model = model_c()
        print(f"{model_name}: {sum(p.numel() for p in model.parameters(recurse=True))}")


def collate_base_results_into_df():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN", "Unet", "InductiveNet", "TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "ensemble" in fname:
            print(fname)
            continue
        if "maximum_consistency" in fname or "last_epoch" in fname:
            print(fname)
            continue

        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            model = fname.split("_")[0]
            data = pickle.load(file)
            datasets, samples = data["ious"].shape

            experiment = "No Augmentation"
            if "sil" in fname and "_G" not in fname:
                experiment = "consistency training"
            elif "_V" in fname:
                experiment = "Vanilla Augmentation"
            elif "_G" in fname:
                experiment = "Inpainter Augmentation"

            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:
                        continue
                    dataset.append([dataset_names[i], model, j, experiment, data["ious"][i, j]])

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "ID", "Experiment", "IoU"])
    # print(iou_dataset)
    dataset = iou_dataset.groupby(["Experiment", "Model", "Dataset"]).head()
    dataset = dataset.reset_index()
    iou_dataset.to_csv("base_data.csv")
    return iou_dataset


def plot_ensemble_performance():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN", "Unet", "InductiveNet", "TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles/ensemble")):

        with open(os.path.join("experiments/Data/pickles/ensemble", fname), "rb") as file:
            model = fname.split("-")[0]
            data = pickle.load(file)
            datasets, samples = data["ious"].shape
            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:
                        continue
                    dataset.append([dataset_names[i], model, data["ious"][i, j]])
    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "IoU"])
    iou_dataset.to_csv("single_model_ensemble_data.csv")

    plot = sns.barplot(x="Dataset", y="IoU", hue="Model", data=iou_dataset)
    plt.show()
    # plt.savefig("ensemble_plot.png")
    return iou_dataset


def plot_inpainter_vs_conventional_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] != "consistency training"]
    filt = df.groupby(["Dataset", "ID", "IoU", "Experiment"]).mean()
    filt = filt.reset_index()
    hue_order = df.groupby(["Experiment"])["IoU"].mean().sort_values().index
    order = df.groupby(["Dataset"])["IoU"].mean().sort_values().index
    table = df.groupby(["Dataset", "Model", "Experiment"])["IoU"].mean()
    test = table.to_latex()
    print(test)
    sns.barplot(data=filt, x="Dataset", y="IoU", hue="Experiment", hue_order=hue_order, order=order)
    plt.show()
    return table


def plot_training_procedure_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"] != "Inpainter Augmentation"]
    filt = df.groupby(["Dataset", "ID", "IoU", "Experiment"]).mean()
    filt = filt.reset_index()
    hue_order = df.groupby(["Experiment"])["IoU"].mean().sort_values().index
    order = df.groupby(["Dataset"])["IoU"].mean().sort_values().index
    table = df.groupby(["Dataset", "Model", "Experiment"])["IoU"].mean()
    test = table.to_latex(float_format="%.3f")
    print(test)
    sns.barplot(data=filt, x="Dataset", y="IoU", hue="Experiment", hue_order=hue_order, order=order)
    plt.show()
    return table


if __name__ == '__main__':
    # def test(a):
    #     return np.mean()
    # get_boxplots_for_models()
    # collate_results_into_df()
    # get_variances_for_models()
    # plot_ensemble_performance()
    # collate_base_results_into_df()
    # plot_parameters_sizes()
    # training_plot("logs/vanilla/DeepLab/vanilla_1.csv")
    # plot_inpainter_vs_conventional_performance()
    plot_training_procedure_performance()
