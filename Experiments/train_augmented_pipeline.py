from training.consistency_trainers import ConsistencyTrainer
import sys

if __name__ == '__main__':
    config = {"model": "DeepLab",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250}
    trainer = ConsistencyTrainer(id=f"{sys.argv[1]}", config=config)
    trainer.train()
