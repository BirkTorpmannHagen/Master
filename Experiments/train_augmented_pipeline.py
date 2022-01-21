from training.consistency_trainers import ConsistencyTrainer, AdversarialConsistencyTrainer, \
    ConsistencyTrainerUsingAugmentation

if __name__ == '__main__':
    config = {"device": "cuda",
              "pretrain": "imagenet",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250}
    # trainer = ContrastiveTrainer(model_str="DeepLab", id="fully_stochastic_weights", config=config)
    # trainer.train()
    for i in range(0, 10):
        trainer = ConsistencyTrainer(model_str="DeepLab", id=f"sanity_check_{i}", config=config)
        # trainer = AdversarialConsistencyTrainer(model_str="DeepLab", id=f"adversarial_{i}_sanity_check", config=config)
        trainer.train()
