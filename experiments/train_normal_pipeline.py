import sys
from collect_generalizability_metrics import get_generalizability_gap
from training.vanilla_trainer import VanillaTrainer

if __name__ == '__main__':
    config = {"model": "DeepLab",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250}
    trainer = VanillaTrainer(sys.argv[1], config)
    trainer.train()

    # for i in ["TriUnet", "DeepLab", "FPN", "Unet"]:
    #     config["model"] = i
    #     trainer = VanillaTrainer(sys.argv[1], config)
    #     trainer.train()
    # for i in range(13, 100):
    #     trainer = VanillaTrainer("DeepLab", i, config)
    #     trainer.train()
    # i = 4
    # while True:
    #     config = {"epochs": 200, "id": i, "lr": 0.00001, "pretrain": False}
    #     training.vanilla_trainer.train_vanilla_predictor(config)
    #     i += 1
