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
from utils import logging


class VanillaTrainer:
    def __init__(self, model_str, id, config):
        """

        :param model: String describing the model type. Can be DeepLab, Divergent, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.device = config["device"]
        self.pretrain = config["pretrain"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.model = None
        self.id = id
        self.model_str = model_str
        if model_str == "DeepLab":
            if self.pretrain == "KvasirClassification":
                self.model = backbones.DeepLab().to(self.device)
                pretrain_config = {"epochs": 100, "lr": 0.000001, "id": id}
                encoder = pretrain_encoder(self.model, KvasirClassificationDataset("Datasets/HyperKvasir"),
                                           config=pretrain_config)
                self.model.encoder.load_state_dict(encoder.state_dict())
            elif self.pretrain == "imagenet":
                self.model = backbones.DeepLab("imagenet").to(self.device)
        elif model_str == "Divergent":
            self.model = backbones.DivergentNet().to(self.device)
            if self.pretrain == "KvasirClassification":
                raise NotImplementedError
            elif "None":
                pass
        elif model_str == "Polyp":
            self.model = backbones.PolypNet().to(self.device)
            if self.pretrain == "KvasirClassification":
                raise NotImplementedError
        else:
            raise AttributeError("model_str not valid; choices are DeepLab, Divergent, Polyp")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = vanilla_losses.JaccardLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25)
        self.dataset = KvasirSegmentationDataset("Datasets/HyperKvasir")
        self.train_set, self.val_set = random_split(self.dataset, [int(len(self.dataset) * 0.8),
                                                                   len(self.dataset) - int(len(self.dataset) * 0.8)])
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def train_epoch(self):
        self.model.train()
        losses = []
        image_aug = aug.image_transforms()
        seg_aug = aug.SegAugmentations(0.5)
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            image = image_aug(image)
            mask = y.to("cuda")
            image, mask = seg_aug(image, mask)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)

    def train(self):
        best_iou = 0
        print("Starting Segmentation training")
        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            validation_loss, ious = self.validate(epoch=i, plot=False)
            logging.log_iou(f"logs/training_log_{self.model_str}_pretrainmode={self.pretrain}_{self.id}", i, ious)
            mean_iou = torch.mean(ious)
            self.scheduler.step(i)
            print(
                f"Epoch {i} of {self.epochs} \t lr={[group['lr'] for group in self.optimizer.param_groups]}, loss={training_loss} \t val_loss={validation_loss}, mean_iou={mean_iou}")

            if mean_iou > best_iou:
                print("Saving best model..")
                best_iou = mean_iou
                np.save(f"Experiments/Normal-Pipelines/{self.model_str}/pretrainmode={self.pretrain}_{self.id}", ious)
                torch.save(self.model.state_dict(),
                           f"Predictors/{self.model_str}/pretrainmode={self.pretrain}_{self.id}")

    def validate(self, epoch, plot=False):
        self.model.eval()
        losses = []
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, fname in self.val_loader:
                image = x.to("cuda")
                mask = y.to("cuda")
                output = self.model(image)
                loss = self.criterion(output, mask)
                losses.append(np.abs(loss.item()))
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                ious = torch.cat((ious, batch_ious.flatten()))
                if plot:
                    # plt.imshow(y[0, 0].cpu().numpy(), alpha=0.5)
                    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
                    plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
                    # plt.imshow(y[0, 0].cpu().numpy().astype(int), alpha=0.5)
                    plt.title("IoU {} at epoch {}".format(iou(output[0, 0], mask[0, 0]), epoch))
                    plt.show()
                    plot = False  # plot one example per epoch
        avg_val_loss = np.mean(losses)
        return avg_val_loss, ious

# def train_epoch(model, training_loader, config):
#     criterion = config["criterion"]
#     device = config["device"]
#     optimizer = config["optimizer"]
#     model.train()
#     losses = []
#     image_aug = aug.image_transforms()
#     seg_aug = aug.SegAugmentations(0.5)
#     for x, y, fname in training_loader:
#         image = x.to("cuda")
#         image = image_aug(image)
#         mask = y.to("cuda")
#         image, mask = seg_aug(image, mask)
#         optimizer.zero_grad()
#         output = model(image)
#         loss = criterion(output, mask)
#         loss.backward()
#         optimizer.step()
#         losses.append(np.abs(loss.item()))
#     return np.mean(losses)
#
#
# def validate(model, validation_loader, config, epoch, plot=False):
#     model.eval()
#     criterion = config["criterion"]
#     device = config["device"]
#     optimizer = config["optimizer"]
#     id = config["id"]
#     batch_size = config["batch_size"]
#     losses = []
#     ious = torch.empty((0,))
#     with torch.no_grad():
#         for x, y, fname in validation_loader:
#             image = x.to("cuda")
#             mask = y.to("cuda")
#             output = model(image)
#             loss = criterion(output, mask)
#             losses.append(np.abs(loss.item()))
#             batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
#             ious = torch.cat((ious, batch_ious.flatten()))
#             if plot:
#                 # plt.imshow(y[0, 0].cpu().numpy(), alpha=0.5)
#                 plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
#                 plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
#                 # plt.imshow(y[0, 0].cpu().numpy().astype(int), alpha=0.5)
#                 plt.title("IoU {} at epoch {}".format(iou(output[0, 0], mask[0, 0]), epoch))
#                 plt.show()
#                 plot = False
#     avg_val_loss = np.mean(losses)
#     return avg_val_loss, ious
#
#
# def train_vanilla_predictor(config):
#     epochs = config["epochs"]
#     id = config["id"]
#     lr = config["lr"]
#     pretrain = config["pretrain"]
#     model = backbones.DeepLab().to("cuda")
#     if pretrain:
#         pretrain_config = {"epochs": 100, "lr": 0.000001, "id": id}
#         encoder = pretrain_encoder(model, KvasirClassificationDataset("Datasets/HyperKvasir"), config=pretrain_config)
#         model.encoder.load_state_dict(encoder.state_dict())  # use pretrained weights
#     model = torch.nn.Sequential(model, torch.nn.Sigmoid())
#     device = torch.cuda.device(torch.cuda.current_device())
#     criterion = vanilla_losses.JaccardLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25)
#     train_config = {"criterion": criterion, "device": device, "optimizer": optimizer, "scheduler": scheduler, "id": id,
#                     "batch_size": 16}
#     dataset = KvasirSegmentationDataset("Datasets/HyperKvasir")
#     train_set, val_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
#     train_loader = DataLoader(train_set, batch_size=train_config["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=train_config["batch_size"], shuffle=True)
#     best_iou = 0
#     print("Starting Segmentation training")
#     for i in range(epochs):
#         training_loss = np.abs(train_epoch(model, train_loader, train_config))
#         validation_loss, ious = validate(model, val_loader, train_config, i, plot=False)
#         if pretrain:
#             logging.log_iou("logs/ious.log", i, id, ious)
#         else:
#             logging.log_iou("logs/no-pretrain-ious.log", i, id, ious)
#         mean_iou = torch.mean(ious)
#         scheduler.step(i)
#         print(
#             "Epoch: {}/{} with lr {} \t Training loss:{}\t validation loss: {}\t IoU: {}".format(i, epochs,
#                                                                                                  [group['lr'] for group
#                                                                                                   in
#                                                                                                   optimizer.param_groups],
#                                                                                                  training_loss,
#                                                                                                  validation_loss,
#                                                                                                  mean_iou))
#
#         if mean_iou > best_iou:
#             print("Saving best model..")
#             best_iou = mean_iou
#             if pretrain:
#                 torch.save(model.state_dict(), "Predictors/BaselineDeepLab/DeepLab-{}".format(id))
#             else:
#                 torch.save(model.state_dict(), "Predictors/BaselineDeepLab/DeepLab-nopretrain-{}".format(id))
