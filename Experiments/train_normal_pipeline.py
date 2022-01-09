from training.vanilla_trainer import VanillaTrainer

if __name__ == '__main__':
    config = {"device": "cuda",
              "pretrain": "imagenet",
              "lr": 0.00001,
              "batch_size": 16,
              "epochs": 250}
    for i in range(1, 5):
        trainer = VanillaTrainer("DeepLab", f"250_epochs_{i}", config)
        trainer.train()
    # for i in range(13, 100):
    #     trainer = VanillaTrainer("DeepLab", i, config)
    #     trainer.train()
    # i = 4
    # while True:
    #     config = {"epochs": 200, "id": i, "lr": 0.00001, "pretrain": False}
    #     training.vanilla_trainer.train_vanilla_predictor(config)
    #     i += 1
