import torch
import torch.nn as nn
import numpy as np
from models import segmentation_models
import segmentation_models_pytorch as smp
from os import listdir
from os.path import join


class SingularEnsemble(nn.Module):
    def __init__(self, model_str, model_count, state_dir, predictor_type=None):
        super(SingularEnsemble, self).__init__()

        state_fnames = listdir(state_dir)
        assert len(
            state_fnames) > model_count, f"Not enough trained instances of {model_str} for ensemble of size {model_count}"
        if predictor_type is None:
            state_fnames = [join(state_dir, str(i)) for i in range(1, model_count + 1)]
        else:
            state_fnames = [i for i in state_fnames if predictor_type in i][
                           :model_count]  # filters according to predictor type, i.e maximum_consistency
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
        for model, fname in zip(self.models, state_fnames):
            model.load_state_dict(torch.load(fname))

    def forward(self, x, threshold=True):
        out = torch.zeros((self.model_count, x.shape[-4], 1, x.shape[-2], x.shape[-1])).to(self.device)
        for i in range(len(self.models)):
            if self.model_str == "InductiveNet":
                out[i], _ = self.models[i](x)
            else:
                out[i] = self.models[i](x)
        if threshold:
            return (torch.mean(out, 0) / self.model_count > 0.5).float()
        return torch.mean(out, 0) / self.model_count


class DiverseEnsemble(nn.Module):
    def __init__(self):
        super(DiverseEnsemble, self).__init__()
        vanilla_models = [segmentation_models.FPN(), segmentation_models.Unet(), segmentation_models.TriUnet(),
                          segmentation_models.DeepLab()]
        inductiveNet_ensemble = segmentation_models.InductiveNet()

    def forward(self, x):
        pass


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


if __name__ == '__main__':
    # ens = SingularEnsemble("InductiveNet", 10, "Predictors/Augmented/InductiveNet")
    ens = TrainedEnsemble("Singular")
    ens(torch.zeros(8, 3, 512, 512).to("cuda"))
    print("Done!")
