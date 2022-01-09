from training.contrastive_trainer import ContrastiveTrainer

if __name__ == '__main__':
    config = {"device": "cuda",
              "pretrain": "imagenet",
              "lr": 0.00001,
              "batch_size": 16,
              "epochs": 250}
    # trainer = ContrastiveTrainer(model_str="DeepLab", id="fully_stochastic_weights", config=config)
    # trainer.train()
    for i in range(2, 10):
        trainer = ContrastiveTrainer(model_str="DeepLab", id=f"new_loss_{i}", config=config)
        trainer.train()
