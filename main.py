import Models.backbones as backbones
import training.vanilla_trainer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    model = backbones.DeepLab(1).to("cuda")
    training.vanilla_trainer.train_vanilla_predictor(model, 1000)
