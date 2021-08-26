import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

from DataProcessing.hyperkvasir import KvasirDataset


def train_epoch(model, training_loader, config):
    criterion, device, optimizer = config.items()
    model.train()
    losses = []
    for x, y, fname in tqdm(training_loader):
        image = x.to("cuda")
        mask = y.to("cuda")
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # print(fname)
        # print(x.shape)
        # print(y.shape)
    return np.mean(losses)


def validate(model, validation_loader):
    print("Validating")
    model.eval()
    with torch.no_grad():
        for x, y, fname in validation_loader:
            loss = 0
            print("Validation score: {}", format(loss))


def train_vanilla_predictor(model, epochs):
    print(torch.cuda.current_device())
    device = torch.cuda.device(torch.cuda.current_device())
    print(device)
    criterion = vanilla_losses.DiceLoss
    print(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    config = {"criterion": criterion, "device": device, "optimizer": optimizer}
    dataset = KvasirDataset("Data/segmented-images/")
    train_set, val_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set))
    for i in range(epochs):
        train_epoch(model, train_loader, config)
        validate(model, val_loader)
