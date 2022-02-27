from training.consistency_trainers import *
from training.inductivenet_trainer import InductiveNetTrainer
import sys

if __name__ == '__main__':
    config = {"model": "InductiveNet",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 250,
              "use_inpainter": False}
    trainer = InductiveNetTrainer(id=f"{sys.argv[1]}", config=config)
    trainer.train()
    config = {"model": "FPN",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250,
              "use_inpainter": False}
    trainer = ConsistencyTrainer(id=f"{sys.argv[1]}", config=config)
    trainer.train()
    # config = {"model": "Unet",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = StrictConsistencyTrainer(id=f"dual_jaccard-{sys.argv[1]}", config=config)
    # trainer.train()

    # trainer = ConsistencyTrainerUsingAugmentation(id=f"augmentation-{sys.argv[1]}", config=config)
    # trainer.train()
    # config = {"model": "Unet",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = ConsistencyTrainerUsingControlledAugmentation("aug_test", config)
    # trainer.train()
    # config = {"model": "Unet",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = AdversarialConsistencyTrainer(id=f"adversarial-{sys.argv[1]}", config=config)
    # trainer.train()
