import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import join


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


if __name__ == '__main__':
    get_predictorwise_distribution("Experiments/Data/Normal-Pipelines/DeepLab")
