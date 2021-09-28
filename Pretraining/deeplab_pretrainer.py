import torch.nn as nn
import torch.optim.optimizer
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torchvision.ops.focal_loss import sigmoid_focal_loss

from DataProcessing.hyperkvasir import KvasirClassificationDataset
from Models.backbones import DeepLab


def pretrain_encoder(seg_model, dataset, config):
    """
    :param model: model
    :param dataset: Dataset object
    :param config: hyperparameter dictionary, {epoch:int, lr:float}
    :return: trained encoder
    """
    print("Pretraining ")
    feature_to_logits = nn.Sequential()
    backbone = resnet34(pretrained=False)
    # todo train then perform the same modifications
    epochs = config["epochs"]
    lr = config["lr"]
    criterion = sigmoid_focal_loss
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)

    for epoch in range(epochs):
        for x, y, fname in DataLoader(dataset, 16, True):
            image = x.to("cuda")
            label = y.to("cuda")
            optimizer.zero_grad()
            output = model(image)[-1]  # todo check; might be wrong
            print(label.shape)
            print(output.shape)
            input()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            print(torch.mean(loss.item()))


if __name__ == '__main__':
    config = {"epochs": 100, "lr": 1e-6}
    dataset = KvasirClassificationDataset("Data")
    model = DeepLab(dataset.num_classes).to("cuda")
    pretrain_encoder(model, dataset, config)
