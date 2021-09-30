from os import listdir

import numpy as np
import torch.optim.optimizer
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.models import resnet34
from torchvision.ops.focal_loss import sigmoid_focal_loss

import utils.logging as logging
from DataProcessing.augmentation import image_transforms
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
    epochs = config["epochs"]
    # augment = AutoAugment()
    augment = image_transforms()
    lr = config["lr"]
    id = config["id"]
    if "Resnet34-{}".format(id) in listdir("Predictors/ResNetBackbones"):
        print("Found pretrained")
    else:
        criterion = sigmoid_focal_loss
        optimizer = torch.optim.Adam(backbone.parameters())
        train_set, val_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25)
        best_bal_acc = 0
        best_model = None
        for epoch in range(epochs):
            losses = []
            backbone.train()
            for i, (x, y, fname) in enumerate(DataLoader(train_set, 4, shuffle=True)):
                image = x.to("cuda")
                label = y.to("cuda")
                # image = augment(ToPILImage()(image).convert("RGB").) #TODO check if this is better
                image = augment(image)
                optimizer.zero_grad()
                output = backbone(image)  # todo check; might be wrong
                loss = criterion(output, label, reduction="mean")
                loss.backward()
                optimizer.step()
                scheduler.step(epoch)
                losses.append(loss.item())

            preds_concat = torch.empty((0, dataset.num_classes))
            label_concat = torch.empty((0, dataset.num_classes))
            backbone.eval()
            for x, y, fname in DataLoader(val_set):
                with torch.no_grad():
                    image = x.to("cuda")
                    label = y.to("cuda")
                    output = backbone(image)
                    preds_concat = torch.vstack((preds_concat, output.cpu()))
                    label_concat = torch.vstack((label_concat, label.cpu()))
            preds_concat = np.argmax(preds_concat.numpy(), axis=1)
            label_concat = np.argmax(label_concat.numpy(), axis=1)
            bal_acc = balanced_accuracy_score(label_concat, preds_concat)
            training_loss = np.mean(
                losses)
            accuracy = accuracy_score(label_concat, preds_concat)
            print("Epoch: {}, \t Training Loss: {} \t Balanced Accuracy: {} \t Actual accuracy: {}".format(epoch,
                                                                                                           training_loss,
                                                                                                           bal_acc,
                                                                                                           accuracy
                                                                                                           ))
            logging.log("logs/pretrain.log", ["id", "epoch", "train_loss", "balanced_acc", "acc"],
                        [id, epoch, training_loss,
                         bal_acc, accuracy])
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                print("Found new best model with accuracy {}!".format(best_bal_acc))
                torch.save(backbone.state_dict(), "Predictors/ResNetBackbones/Resnet34-{}".format(id))
    backbone.load_state_dict(torch.load("Predictors/ResNetBackbones/Resnet34-{}".format(id)))
    # convert to encoder
    del backbone.fc
    del backbone.avgpool
    return backbone


if __name__ == '__main__':
    config = {"epochs": 100, "lr": 0.000001, "id": 1}
    dataset = KvasirClassificationDataset("Data")
    model = DeepLab(dataset.num_classes).to("cuda")
    pretrain_encoder(model, dataset, config)
