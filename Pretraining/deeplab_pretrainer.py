from DataProcessing import


def pretrain_encoder(self, dataset, config):
    """

    :param dataset: string specifying location of Hyperkvasir labeled-images dataset
    :param config: hyperparameter dictionary, {epoch:int, lr:float, scheduler:object, optimizer:object, criterion: object}
    :return: trained encoder
    """
    print("Pretraining ")
    model = self.encoder.model
    epochs = config["epochs"]
    criterion = config["criterion"]
    lr = config["lr"]
    scheduler = config["scheduler"]
    for epoch in range(epochs):
        for idx, x, y in enumerate()
