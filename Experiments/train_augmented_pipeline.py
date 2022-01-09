from training.contrastive_trainer import ContrastiveTrainer

if __name__ == '__main__':
    config = {"device": "cuda",
              "pretrain": "imagenet",
              "lr": 0.00001,
              "batch_size": 16,
              "epochs": 500}

    trainer = ContrastiveTrainer(model_str="DeepLab", id="adaptive", config=config)
    trainer.train()
