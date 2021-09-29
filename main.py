import training.vanilla_trainer

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
if __name__ == '__main__':
    i = 2
    while True:
        config = {"epochs": 100, "id": i, "lr": 0.00001}
        training.vanilla_trainer.train_vanilla_predictor(config)
        i += 1
