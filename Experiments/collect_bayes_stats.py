from itertools import combinations

import numpy as np
import torch

from Models.backbones import *


def get_avg_dist_hist(model1, model2, filenames):
    distance_matrix = np.zeros((len(filenames), len(filenames)))
    with torch.no_grad():
        for idx_pair in combinations(range(len(filenames)), 2):
            model1.load_state_dict(torch.load(filenames[idx_pair[0]]))
            model2.load_state_dict(torch.load(filenames[idx_pair[1]]))
            dists = []
            for (param1, param2) in zip(model1.parameters(), model2.parameters()):
                dists.append(np.abs(torch.sum((param1 - param2))))
            print(np.mean(dists))


if __name__ == '__main__':
    get_avg_dist_hist(DeepLab(), DeepLab(),
                      ["Predictors/Augmented/DeepLab/pretrainmode=imagenet_{}".format(i) for i in range(3)])
    print("vanilla:")
    get_avg_dist_hist(DeepLab(), DeepLab(),
                      ["Predictors/Vanilla/DeepLab/pretrainmode=imagenet_250_epochs_{}".format(i) for i in range(4)])
