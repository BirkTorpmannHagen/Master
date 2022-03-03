from training.consistency_trainers import *
from training.vanilla_trainer import *
import sys

if __name__ == '__main__':
    id = sys.argv[1]
    model = sys.argv[2]

    """
    Consistency Training
    """
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 250,
              "use_inpainter": False}
    trainer = ConsistencyTrainer(id, config)
    trainer.train()
    """
    Model-based augmentations
    """
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 250,
              "use_inpainter": False}
    trainer = ConsistencyTrainerUsingAugmentation(id, config)
    trainer.train()
    """
        No augmentations
    """
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 250,
              "use_inpainter": False}
    trainer = VanillaTrainer(id, config)
    trainer.train()
