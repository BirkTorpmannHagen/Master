import training.vanilla_trainer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    i = 0
    while True:
        config = {"epochs": 200, "id": i, "lr": 0.000001, "pretrain": True}
        training.vanilla_trainer.train_vanilla_predictor(config)
        i += 1
