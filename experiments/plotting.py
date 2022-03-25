import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle


def training_plot(log_csv):
    log_df = pd.read_csv(log_csv)
    plt.title("Training Plot Sample")
    plt.xlabel("Epochs")
    plt.ylabel("SIL")
    plt.xlim((0, 500))
    plt.ylim((0, 1))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Training Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Validation Loss")
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
    datadict = {}
    for fname in os.listdir("experiments/Data/pickles"):
        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            data = pickle.load(file)
            datadict = {**datadict, **{"Model": fname.split("_")[0]}}
            print(fname, ": ", data["ious"])
    sns.boxplot(x="Model", y="IoUs", hue="dataset", data=)


if __name__ == '__main__':
    for fname in os.listdir("experiments/Data/pickles"):
        with open(os.path.join("experiments/Data/pickles", fname), "rb") as file:
            data = pickle.load(file)

            print(fname, ": ", data["ious"])
