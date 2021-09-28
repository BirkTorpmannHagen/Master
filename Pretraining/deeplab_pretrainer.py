import numpy as np
import torch.optim.optimizer
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm

from DataProcessing.hyperkvasir import KvasirClassificationDataset
from Models.backbones import DeepLab


def pretrain_encoder(seg_model, dataset, config):
    """
    :param seq_model: model
    :param dataset: Dataset object
    :param config: hyperparameter dictionary, {epoch:int, lr:float}
    :return: trained encoder
    """
    print("Pretraining ")
    backbone = resnet34(pretrained=False, num_classes=dataset.num_classes).to("cuda")
    # TODO delete last two layers to convert to encoder
    epochs = config["epochs"]
    lr = config["lr"]
    criterion = sigmoid_focal_loss
    # training staghe
    backbone.train()
    optimizer = torch.optim.Adam(backbone.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100)
    prev_params = backbone.parameters()
    for epoch in tqdm(range(epochs)):
        losses = []
        for x, y, fname in DataLoader(dataset, 1, shuffle=True):
            image = x.to("cuda")
            label = y.to("cuda")
            optimizer.zero_grad()
            output = backbone(image)  # todo check; might be wrong
            loss = criterion(output, label, reduction="mean")
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            losses.append(loss.item())
        print(np.mean(losses))


if __name__ == '__main__':
    config = {"epochs": 100, "lr": 0.00001}
    dataset = KvasirClassificationDataset("Data")
    model = DeepLab(dataset.num_classes).to("cuda")
    pretrain_encoder(model, dataset, config)
