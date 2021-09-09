import torch.nn

import Models.backbones as backbones
import training.vanilla_trainer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    i = 1
    while True:
        model = backbones.DeepLab(1).to("cuda")
        model = torch.nn.Sequential(model, torch.nn.Sigmoid())
        training.vanilla_trainer.train_vanilla_predictor(model, 200, i)
        i += 1
# TODO modify model to include activation
