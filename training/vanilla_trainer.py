import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
import DataProcessing.augmentation as aug
from DataProcessing.hyperkvasir import KvasirSegmentationDataset, KvasirClassificationDataset
from Models import backbones
from Models import inpainters
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
            elif self.pretrain == "discriminator":  # for experiment
                print("disc")
                # self.model = backbones.DeepLab().to(self.device)
                # self.model = inpainters.SegDiscriminator().to(self.device)
                # self.model.load_state_dict(torch.load("Predictors/Inpainters/better-deeplab-discriminator-990"))
                self.gen = inpainters.SegGenerator()
                self.gen.load_state_dict(torch.load("Predictors/Inpainters/no-pretrain-deeplab-generator-4990"))
                self.model = torch.nn.Sequential(self.gen,
                                                 torch.nn.Conv1d(3, 1, (1, 1)),
                                                 torch.nn.Sigmoid()).to(self.device)
        elif model_str == "Divergent":
            self.model = backbones.DivergentNet().to(self.device)
            if self.pretrain == "KvasirClassification":
                raise NotImplementedError
            elif "None":
                pass
        elif model_str == "Polyp":
            raise NotImplementedError
        elif model_str == "Unet":
            raise NotImplementedError
        elif model_str == "FPN":
            raise NotImplementedError
        else:
            raise AttributeError("model_str not valid; choices are DeepLab, Divergent, Polyp, FPN, Unet")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = vanilla_losses.JaccardLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25)
        self.train_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="train")
        self.val_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="val")
        self.test_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="test")
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
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
            validation_loss, ious = self.validate(epoch=i, plot=True)
            test_ious = self.test()
            logging.log_iou(f"logs/training_log_{self.model_str}_pretrainmode={self.pretrain}_{self.id}", i, ious)
            mean_iou = torch.mean(ious)
            self.scheduler.step(i)
            print(
                f"Epoch {i} of {self.epochs} \t lr={[group['lr'] for group in self.optimizer.param_groups]}, loss={training_loss} \t val_loss={validation_loss}, mean_iou={mean_iou}")

            if mean_iou > best_iou:
                best_iou = mean_iou
                np.save(f"Experiments/Data/Normal-Pipelines/{self.model_str}/pretrainmode={self.pretrain}_{self.id}",
                        test_ious)

                print(f"Saving new best model. Test-set mean iou: {float(np.mean(test_ious.numpy()))}")
                torch.save(self.model.state_dict(),
                           f"Predictors/{self.model_str}/pretrainmode={self.pretrain}_{self.id}")

    def test(self):
        self.model.eval()
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, fname in self.test_loader:
                image = x.to("cuda")
                mask = y.to("cuda")
                output = self.model(image)
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                ious = torch.cat((ious, batch_ious.flatten()))
        return ious

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
