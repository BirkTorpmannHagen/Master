from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_predictorwise_distribution(experiment_path):
    mean_ious = []
    for experiment in listdir(experiment_path):
        # mean_ious.append(np.load(join(experiment_path, experiment))) #cool
        mean_ious.append(np.mean(np.load(join(experiment_path, experiment))))
    plt.hist(mean_ious, bins=np.linspace(min(mean_ious), max(mean_ious), 25))
    plt.show()


def get_datawise_distribution(experiment_path):
    mean_ious = []
    for experiment in listdir(experiment_path):
        # mean_ious.append(np.load(join(experiment_path, experiment))) #cool
        mean_ious = np.concatenate((mean_ious, np.mean(np.load(join(experiment_path, experiment)))))
        plt.hist(mean_ious, bins=np.linspace(min(mean_ious), max(mean_ious), 25))
        plt.show()


def plot_training_progression(csv_name):
    df = pd.read_csv(csv_name)
    plt.plot(df["epoch"], df["iid_test_iou"], label="IID")
    plt.plot(df["epoch"], df["ood_iou"], label="OOD")
    plt.plot(df["epoch"], df["iid_test_iou"] - df["ood_iou"], label="Diff")
    plt.legend()
    plt.ylim((0, 1))
    plt.xlim((0, 250))
    plt.show()


if __name__ == '__main__':
    plot_training_progression("logs/consistency/DeepLab/0.csv")
    plot_training_progression("logs/consistency/DeepLab/1.csv")
    # df_iid = df["Augmented" not in df["name"]]
    # get_predictorwise_distribution("Experiments/Data/Normal-Pipelines/DeepLab")
