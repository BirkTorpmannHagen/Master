import training.vanilla_trainer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    i = 4
    while True:
        config = {"epochs": 200, "id": i, "lr": 0.00001, "pretrain": False}
        training.vanilla_trainer.train_vanilla_predictor(config)
        i += 1
