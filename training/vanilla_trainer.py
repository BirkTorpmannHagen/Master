import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from DataProcessing.hyperkvasir import KvasirDataset
from Tests.metrics import iou


def train_epoch(model, training_loader, config):
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    model.train()
    losses = []
    for x, y, fname in training_loader:
        image = x.to("cuda")
        mask = y.to("cuda")
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        losses.append(np.abs(loss.item()))
    return np.mean(losses)


def validate(model, validation_loader, config, plot=False):
    model.eval()
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    losses = []
    ious = []
    with torch.no_grad():
        for x, y, fname in validation_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            output = model(image)
            loss = criterion(output, mask)
            losses.append(np.abs(loss.item()))
            batch_ious = [iou(output_i, mask_j).cpu().numpy() for output_i, mask_j in zip(output, mask)]
            ious.append(np.mean(batch_ious))
            # print([float(i) for i in batch_ious])
            if plot:
                # plt.imshow(y[0, 0].cpu().numpy(), alpha=0.5)
                # plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
                plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
                plt.imshow(y[0, 0].cpu().numpy().astype(int), alpha=0.5)
                plt.title(iou(output[0, 0], mask[0, 0]))
                plt.show()
                plot = False
    avg_val_loss = np.mean(losses)
    avg_iou = np.mean(ious)
    return avg_val_loss, avg_iou


def train_vanilla_predictor(model, epochs, num):
    device = torch.cuda.device(torch.cuda.current_device())
    # criterion = vanilla_losses.BCEWithLogitsLoss()
    criterion = vanilla_losses.JaccardLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    optimizer = torch.optim.Adam(model.parameters(), 0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    config = {"criterion": criterion, "device": device, "optimizer": optimizer, "scheduler": scheduler}
    dataset = KvasirDataset("Data/segmented-images/")
    train_set, val_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    best_validation_loss = 10
    for i in range(epochs):
        training_loss = np.abs(train_epoch(model, train_loader, config))
        validation_loss, iou = validate(model, val_loader, config)
        scheduler.step(validation_loss)
        print(
            "Epoch: {}/{} with lr {} \t Training loss:{}\t validation loss: {}\t IoU: {}".format(i, epochs,
                                                                                                 [group['lr'] for group
                                                                                                  in
                                                                                                  optimizer.param_groups],
                                                                                                 training_loss,
                                                                                                 validation_loss, iou))
        if validation_loss < best_validation_loss:
            print("Saving best model..")
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), "Models/Predictors/{}-{}".format(type(model).__name__,
                                                                            num))
