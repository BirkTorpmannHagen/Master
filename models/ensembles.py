import random

import torch
import torch.nn as nn
import numpy as np
from models import segmentation_models
import segmentation_models_pytorch as smp
from os import listdir
from os.path import join
import re


class SingularEnsemble:
    def __init__(self, model_str, state_dir, predictor_type, model_count=4, random_choice=True):
        # Ensemble consisting of only one type of Predictor
        self.state_fnames = listdir(state_dir)
        assert len(
            self.state_fnames) > model_count, f"Not enough trained instances of {model_str} for ensemble of size {model_count}"
        self.state_fnames = [join(state_dir, i) for i in
                             list(filter(re.compile(f"^{predictor_type}_\d$").search, self.state_fnames))]
        if random_choice:
            random.shuffle(self.state_fnames)
        else:
            self.state_fnames.sort()
        self.state_fnames = self.state_fnames[:model_count]

        self.device = "cuda"
        self.model_str = model_str
        self.model_count = model_count
        if self.model_str == "DeepLab":
            self.models = [segmentation_models.DeepLab().to(self.device).eval() for _ in range(model_count)]
        elif self.model_str == "TriUnet":
            self.models = [segmentation_models.TriUnet().to(self.device).eval() for _ in range(model_count)]
        elif self.model_str == "Unet":
            self.models = [segmentation_models.Unet().to(self.device).eval() for _ in range(model_count)]
        elif self.model_str == "FPN":
            self.models = [segmentation_models.FPN().to(self.device).eval() for _ in range(model_count)]
        elif self.model_str == "InductiveNet":
            self.models = [segmentation_models.InductiveNet().to(self.device).eval() for _ in range(model_count)]
        else:
            raise AttributeError("model_str not valid; choices are DeepLab, TriUnet, FPN, Unet, InductiveNet")
        for model, fname in zip(self.models, self.state_fnames):
            model.load_state_dict(torch.load(fname))

    def predict(self, x, threshold=True):
        out = torch.zeros((self.model_count, x.shape[-4], 1, x.shape[-2], x.shape[-1])).to(self.device)
        for i in range(len(self.models)):
            out[i] = self.models[i].predict(x)
        if threshold:
            return (torch.mean(out, 0) > 0.5).float()
        return torch.mean(out, 0)

    def get_constituents(self):
        return self.state_fnames

    def __call__(self, x, threshold=True):
        return self.predict(x, threshold)


class TrainedEnsemble(nn.Module):
    def __init__(self, ensemble_type):
        super(TrainedEnsemble, self).__init__()
        self.device = "cuda"
        self.consensus_network = smp.DeepLabV3Plus(in_channels=4, classes=1, activation="sigmoid").to(self.device)
        if ensemble_type == "Singular":
            self.ensemble = SingularEnsemble("InductiveNet", 10, "Predictors/Augmented/InductiveNet").to(self.device)
        else:
            print("diverse")
            self.ensemble = DiverseEnsemble().to(self.device)

    def forward(self, x):
        with torch.no_grad():
            mask_avg = self.ensemble(x)
        img = self.consensus_network(torch.cat((mask_avg, x), 1))  # channelwise concatenation
        return img


class DiverseEnsemble:
    def __init__(self, id, state_dir, type):
        self.device = "cuda"
        self.model_names = ["DeepLab", "Unet", "TriUnet", "FPN", "InductiveNet"]
        self.model_constructors = [segmentation_models.DeepLab, segmentation_models.Unet, segmentation_models.TriUnet,
                                   segmentation_models.FPN, segmentation_models.InductiveNet]
        self.models = [i().to(self.device).eval() for i in self.model_constructors]
        for model_name, model in zip(self.model_names, self.models):
            model.load_state_dict(torch.load(f"{join(state_dir, model_name, type)}_{id}"))
        self.id = id

    def get_constituents(self):
        return self.id

    def predict(self, x, threshold=True):
        out = torch.zeros((len(self.models), x.shape[-4], 1, x.shape[-2], x.shape[-1])).to(self.device)
        for i in range(len(self.models)):
            out[i] = self.models[i].predict(x)
        if threshold:
            return (torch.mean(out, 0) > 0.5).float()
        return torch.mean(out, 0)


if __name__ == '__main__':
    # ens = SingularEnsemble("InductiveNet", 10, "Predictors/Augmented/InductiveNet")
    # ens = SingularEnsemble("DeepLab", "Predictors/Augmented/DeepLab", predictor_type="consistency")
    ens = DiverseEnsemble(1, state_dir="Predictors/Augmented", type="consistency")
    ens.predict(torch.zeros(8, 3, 512, 512).to("cuda"))
    print("Done!")
