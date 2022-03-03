from training.consistency_trainers import *
from training.vanilla_trainer import *
import sys

if __name__ == '__main__':
    config = {"model": "InductiveNet",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 4,
              "epochs": 250,
              "use_inpainter": False}
    trainer = EnsembleConsistencyTrainer("ensemble", config)
    trainer.train()
