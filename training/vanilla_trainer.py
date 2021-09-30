import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import DataProcessing.augmentation as aug
from DataProcessing.hyperkvasir import KvasirSegmentationDataset, KvasirClassificationDataset
from Models import backbones
from Pretraining.deeplab_pretrainer import pretrain_encoder
from Tests.metrics import iou


def train_epoch(model, training_loader, config):
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    model.train()
    losses = []
    image_aug = aug.image_transforms()
    seg_aug = aug.seg_augmentations
    for x, y, fname in training_loader:
        image = x.to("cuda")
        image = image_aug(image)
        mask = y.to("cuda")
        image, mask = seg_aug(0.5)(image, mask)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        losses.append(np.abs(loss.item()))
    return np.mean(losses)


def validate(model, validation_loader, config, epoch, plot=False):
    model.eval()
    criterion = config["criterion"]
    device = config["device"]
    optimizer = config["optimizer"]
    id = config["id"]
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
                plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
                plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
                # plt.imshow(y[0, 0].cpu().numpy().astype(int), alpha=0.5)
                plt.title("IoU {} at epoch {}".format(iou(output[0, 0], mask[0, 0]), epoch))
                plt.show()
                plot = False
    avg_val_loss = np.mean(losses)
    avg_iou = np.mean(ious)
    return avg_val_loss, avg_iou


def train_vanilla_predictor(config):
    epochs = config["epochs"]
    id = config["id"]
    lr = config["lr"]
    pretrain = config["pretrain"]
    model = backbones.DeepLab(1).to("cuda")
    if pretrain:
        pretrain_config = {"epochs": 100, "lr": 0.000001, "id": id}
        encoder = pretrain_encoder(model, KvasirClassificationDataset("Data"), config=pretrain_config)
        model.encoder.load_state_dict(encoder.state_dict())  # use pretrained weights
    model = torch.nn.Sequential(model, torch.nn.Sigmoid())
    device = torch.cuda.device(torch.cuda.current_device())
    criterion = vanilla_losses.JaccardLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25)
    train_config = {"criterion": criterion, "device": device, "optimizer": optimizer, "scheduler": scheduler, "id": id}
    dataset = KvasirSegmentationDataset("Data/")
    train_set, val_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
    best_iou = 0
    print("Starting Segmentation training")
    for i in range(epochs):
        training_loss = np.abs(train_epoch(model, train_loader, train_config))
        validation_loss, iou = validate(model, val_loader, train_config, i, plot=True)
        scheduler.step(i)
        # scheduler.step(validation_loss)
        print(
            "Epoch: {}/{} with lr {} \t Training loss:{}\t validation loss: {}\t IoU: {}".format(i, epochs,
                                                                                                 [group['lr'] for group
                                                                                                  in
                                                                                                  optimizer.param_groups],
                                                                                                 training_loss,
                                                                                                 validation_loss, iou))
        if iou > best_iou:
            print("Saving best model..")
            best_iou = iou
            torch.save(model.state_dict(), "Predictors/BaselineDeepLab/DeepLab-{}".format(id))
