from training.consistency_trainers import *
import sys

if __name__ == '__main__':
    config = {"model": "DeepLab",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250,
              "use_inpainter": False}
    # config["model"] = sys.argv[2]
    trainer = ConsistencyTrainer(id=f"{sys.argv[1]}", config=config)
    trainer.train()
    # config = {"model": "DeepLab",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = StrictConsistencyTrainer(id=f"dual_jaccard-{sys.argv[1]}", config=config)
    # trainer.train()
    # trainer = ConsistencyTrainerUsingAugmentation(id=f"augmentation-{sys.argv[1]}", config=config)
    # trainer.train()
    # trainer = ConsistencyTrainerUsingControlledAugmentation("aug_test", config)
    # trainer.train()
    # config = {"model": "DeepLab",
    #           "device": "cuda",
    #           "lr": 0.00001,
    #           "batch_size": 8,
    #           "epochs": 250,
    #           "use_inpainter": False}
    # trainer = AdversarialConsistencyTrainer(id=f"adversarial-{sys.argv[1]}", config=config)
    # trainer.train()
