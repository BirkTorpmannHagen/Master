import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
from utils.formatting import SafeDict
from scipy.stats import wasserstein_distance
from scipy.stats import ttest_ind
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

def collate_ensemble_results_into_df():
    dataset_names = ["Kvasir-SEG", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
    model_names = ["DeepLab", "FPN", "Unet", "InductiveNet", "TriUnet"]
    dataset = []
    for fname in sorted(os.listdir("experiments/Data/pickles")):
        if "ensemble" not in fname:
            continue
        if "maximum_consistency" in fname or "last_epoch" in fname:
            continue

        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            model = fname.split("-")[0]
            experiment = fname.split("-")[-1]
            data = pickle.load(file)
            datasets, samples = data["ious"].shape
            if model == "InductiveNet":
                model = "DD-DeepLabV3+"
            for i in range(datasets):
                for j in range(samples):
                    if data["ious"][0, j] < 0.75:
                        continue
                    dataset.append([dataset_names[i], model, j, experiment, data["ious"][i, j]])

    iou_dataset = pd.DataFrame(data=dataset, columns=["Dataset", "Model", "ID", "Experiment", "IoU"])
    # print(iou_dataset)
    iou_dataset.to_csv("base_data.csv")
    return iou_dataset

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
            if model == "InductiveNet":
                model = "DD-DeepLabV3+"
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
    iou_dataset.to_csv("base_data.csv")
    return iou_dataset


def plot_ensemble_performance():
    df = collate_ensemble_results_into_df()
    print(df)
    latex = df.groupby(["Model", "Dataset"])["IoU"].mean()
    print(latex.reset_index())
    print(latex.to_latex(float_format="%.3f"))
    order = df.groupby(["Dataset", "Model"])["IoU"].mean().sort_values().index
    sns.barplot(data=df, x="Dataset", y="IoU", hue="Model")
    plt.show()
    grouped_mean = df.groupby(["Dataset", "Model"])["IoU"].mean()
    # print(grouped_mean)
    grouped_iid = np.abs(grouped_mean - grouped_mean["Kvasir-SEG"])/grouped_mean["Kvasir-SEG"]
    # print(grouped_iid)

    nedf = collate_base_results_into_df()
    ne_grouped_mean = nedf.groupby(["Dataset", "Model"])["IoU"].mean()
    # print(ne_grouped_mean)
    ne_grouped_iid = np.abs(ne_grouped_mean["Kvasir-SEG"]-ne_grouped_mean) / ne_grouped_mean["Kvasir-SEG"]
    # print(ne_grouped_iid)

    comparison = ne_grouped_iid-grouped_iid
    comparison = comparison.reset_index()

    sns.barplot(data=comparison, x="Dataset", y="IoU", hue="Model")
    plt.show()

    #plot delta vs variance
    ne_grouped_coeff_std = nedf.groupby(["Dataset", "Model"])["IoU"].std()/ne_grouped_mean
    ne_grouped_coeff_std = ne_grouped_coeff_std.reset_index()
    ne_grouped_coeff_std = ne_grouped_coeff_std.rename(columns={"IoU":"Coeff. StD of IoUs"})
    # print(ne_grouped_coeff_std.head(10))
    sns.barplot(data=ne_grouped_coeff_std, x="Dataset", y="Coeff. StD of IoUs", hue="Model")
    plt.show()
    test = pd.merge(ne_grouped_coeff_std, comparison)
    test=test.rename(columns={"IoU":"Change in Generalizability Gap"})
    # print(test)

    sns.scatterplot(test["Coeff. StD of IoUs"], test["Change in Generalizability Gap"], hue=test["Model"])
    plt.show()

    reduced_dataset = ne_grouped_coeff_std.groupby("Model")["Coeff. StD of IoUs"].mean()
    reduced_dataset = reduced_dataset.reset_index()
    comparison = 100*comparison.groupby("Model")["IoU"].mean()
    comparison = comparison.reset_index()
    dataset = pd.merge(reduced_dataset, comparison)
    dataset=dataset.rename(columns={"IoU":"\u0394%IoU"})

    sns.lineplot(data=dataset, x="Coeff. StD of IoUs", y="\u0394%IoU", color="gray", linestyle='--')
    sns.scatterplot(data=dataset, x="Coeff. StD of IoUs", y="\u0394%IoU", hue=reduced_dataset["Model"], s=100)

    plt.show()

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

def plot_baseline_performance():
    df = collate_base_results_into_df()
    df = df[df["Experiment"]=="No Augmentation"]
    df_van =df.groupby(["Dataset", "Model"])["IoU"].mean()
    df_van = df_van.reset_index()
    # hue_order = df_van.groupby(["Model"])["IoU"].mean().sort_values().index
    order = df_van.groupby(["Dataset"])["IoU"].mean().sort_values().index
    print(df_van)
    #t tests here
    plt.hist(df[df["Dataset"]=="Kvasir-SEG"]["IoU"])
    plt.show()
    sns.barplot(data=df, x="Dataset", y="IoU", hue="Model", order=order)
    plt.show()


if __name__ == '__main__':
    # def test(a):
    #     return np.mean()
    get_boxplots_for_models()
    # collate_results_into_df()
    # get_variances_for_models()
    # plot_ensemble_performance()
    # collate_base_results_into_df()
    # plot_parameters_sizes()
    # training_plot("logs/vanilla/DeepLab/vanilla_1.csv")
    # plot_inpainter_vs_conventional_performance()
    # plot_training_procedure_performance()
    # plot_ensemble_performance()
    # plot_baseline_performance()