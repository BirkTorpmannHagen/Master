from training.vanilla_trainer import VanillaTrainer

if __name__ == '__main__':
    config = {"device": "cuda",
              "pretrain": "imagenet",
              "lr": 0.00001,
              "batch_size": 16,
              "epochs": 1000}
    trainer = VanillaTrainer("DeepLab", "1000_epochs_2", config)
    trainer.train()
    # for i in range(13, 100):
    #     trainer = VanillaTrainer("DeepLab", i, config)
    #     trainer.train()
    # i = 4
    # while True:
    #     config = {"epochs": 200, "id": i, "lr": 0.00001, "pretrain": False}
    #     training.vanilla_trainer.train_vanilla_predictor(config)
    #     i += 1
