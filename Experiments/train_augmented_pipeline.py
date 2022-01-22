from training.consistency_trainers import ConsistencyTrainer

if __name__ == '__main__':
    config = {"model": "DeepLab",
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 250}
    for i in range(0, 10):
        trainer = ConsistencyTrainer(id=f"{i}", config=config)
        # trainer = AdversarialConsistencyTrainer(model_str="DeepLab", id=f"adversarial_{i}_sanity_check", config=config)
        trainer.train()
