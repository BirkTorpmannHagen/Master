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
              "batch_size": 8,
              "epochs": 300,
              "use_inpainter": True}

    if model == "InductiveNet":
        trainer = InductiveNetAugmentationTrainer(f"inpainter_zaugmentation_{id}", config.copy())
        trainer.train()

        # trainer = InductiveNetConsistencyTrainer(f"inpainter_consistency_{id}", config.copy())
        # trainer.train()

    else:
        """
        Consistency Training
        """
        # trainer = ConsistencyTrainer(f"inpainter_consistency_{id}", config.copy())
        # trainer.train()
        """
        Model-based augmentations
        """
        trainer = ConsistencyTrainerUsingAugmentation(f"inpainter_augmentation_{id}", config.copy())
        trainer.train()
