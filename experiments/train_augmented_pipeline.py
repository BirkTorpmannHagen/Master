from training.consistency_trainers import *
from training.inductivenet_trainers import InductiveNetConsistencyTrainer, InductiveNetEnsembleTrainer
import sys

if __name__ == '__main__':
    id = sys.argv[1]
    model = sys.argv[2]
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250,
              "use_inpainter": False}
    trainer = ConsistencyTrainer(id, config)
    trainer.train()

    # config = {"model": "FPN",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = ConsistencyTrainer(id=f"{sys.argv[1]}", config=config)
    # trainer.train()
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
