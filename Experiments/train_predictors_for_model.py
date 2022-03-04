from training.inductivenet_trainers import *
from training.consistency_trainers import *
from training.vanilla_trainer import *
import sys

if __name__ == '__main__':
    id = sys.argv[1]
    model = sys.argv[2]
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 2,
              "use_inpainter": False}

    if model == "InductiveNet":
        trainer = InductiveNetAugmentationTrainer(f"augmentation_{id}", config.copy())
        trainer.train()

        trainer = InductiveNetConsistencyTrainer(f"consistency_{id}", config.copy())
        trainer.train()

        trainer = InductiveNetVanillaTrainer(f"vanilla_{id}", config.copy())
        trainer.train()

    else:
        """
        Consistency Training
        """
        trainer = ConsistencyTrainer(f"consistency_{id}", config.copy())
        trainer.train()
        """
        Model-based augmentations
        """
        trainer = ConsistencyTrainerUsingAugmentation(f"augmentation_{id}", config.copy())
        trainer.train()
        """
            No augmentations
        """
        trainer = VanillaTrainer(f"vanilla_{id}", config.copy())
        trainer.train()
