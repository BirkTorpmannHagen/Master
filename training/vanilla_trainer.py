import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader

from data.etis import EtisDataset
from data.hyperkvasir import KvasirSegmentationDataset
from models import segmentation_models
from evaluation.metrics import iou
from losses.consistency_losses import NakedConsistencyLoss, ConsistencyLoss
from perturbation.model import ModelOfNaturalVariation
from utils import logging


class VanillaTrainer:
    def __init__(self, id, config):
        """

        :param model: String describing the model type. Can be DeepLab, TriUnet, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.config = config
        self.device = config["device"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.model = None
        self.id = id
        self.model_str = config["model"]
        self.mnv = ModelOfNaturalVariation(T0=1).to(self.device)
        self.nakedcloss = NakedConsistencyLoss()
        self.closs = ConsistencyLoss()

        if self.model_str == "DeepLab":
            self.model = segmentation_models.DeepLab().to(self.device)
        elif self.model_str == "TriUnet":
            self.model = segmentation_models.TriUnet().to(self.device)
        elif self.model_str == "Unet":
            self.model = segmentation_models.Unet().to(self.device)
        elif self.model_str == "FPN":
            self.model = segmentation_models.FPN().to(self.device)
        elif self.model_str == "InductiveNet":
            self.model = segmentation_models.InductiveNet().to(self.device)

        else:
            raise AttributeError("model_str not valid; choices are DeepLab, TriUnet, InductiveNet, FPN, Unet")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = vanilla_losses.JaccardLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.train_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="train", augment=False)
        self.val_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="val", augment=False)
        self.test_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="test", augment=False)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set)
        self.test_loader = DataLoader(self.test_set)

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
        best_val_loss = 10
        print("Starting Segmentation training")
        best_closs = 100

        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            val_loss, ious, closs = self.validate(epoch=i, plot=False)
            gen_ious = self.validate_generalizability(epoch=i, plot=False)
            mean_iou = float(torch.mean(ious))
            gen_iou = float(torch.mean(gen_ious))
            consistency = 1 - np.mean(closs)
            test_ious = np.mean(self.test().numpy())
            self.config["lr"] = [group['lr'] for group in self.optimizer.param_groups]
            logging.log_full(epoch=i, id=self.id, config=self.config, result_dict=
            {"train_loss": training_loss, "val_loss": val_loss,
             "iid_val_iou": mean_iou, "iid_test_iou": test_ious, "ood_iou": gen_iou,
             "consistency": consistency}, type="vanilla")

            self.scheduler.step(i)
            print(
                f"Epoch {i} of {self.epochs} \t"
                f" lr={[group['lr'] for group in self.optimizer.param_groups]} \t"
                f" loss={training_loss} \t"
                f" val_loss={val_loss} \t"
                f" ood_iou={gen_iou}\t"
                f" val_iou={mean_iou} \t"
                f" gen_prop={gen_iou / mean_iou}"
            )
            if val_loss < best_val_loss:
                test_ious = self.test()
                best_val_loss = val_loss
                np.save(
                    f"Experiments/Data/Normal-Pipelines/{self.model_str}/{self.id}",
                    test_ious)
                print(f"Saving new best model. IID test-set mean iou: {float(np.mean(test_ious.numpy()))}")
                torch.save(self.model.state_dict(),
                           f"Predictors/Vanilla/{self.model_str}/{self.id}")
                print("saved in: ", f"Predictors/Vanilla/{self.model_str}/{self.id}")
            if closs < best_closs:
                best_closs = closs
                torch.save(self.model.state_dict(),
                           f"Predictors/Vanilla/{self.model_str}/{self.id}-maximum-consistency")
        torch.save(self.model.state_dict(),
                   f"Predictors/Vanilla/{self.model_str}/{self.id}_last_epoch")

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
        closses = []
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, fname in self.val_loader:
                image = x.to("cuda")
                mask = y.to("cuda")
                aug_img, aug_mask = self.mnv(image, mask)
                output = self.model(image)
                aug_output = self.model(aug_img)  # todo consider train on augmented vs non-augmented?

                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                loss = self.closs(aug_mask, mask, aug_output, output, torch.mean(batch_ious))
                losses.append(np.abs(loss.item()))
                closses.append(self.nakedcloss(aug_mask, mask, aug_output, output).item())
                ious = torch.cat((ious, batch_ious.cpu().flatten()))
                if plot:
                    plt.imshow(y[0, 0].cpu().numpy(), alpha=0.5)
                    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
                    plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
                    plt.imshow(y[0, 0].cpu().numpy().astype(int), alpha=0.5)
                    plt.title("IoU {} at epoch {}".format(iou(output[0, 0], mask[0, 0]), epoch))
                    plt.show()
                    plot = False  # plot one example per epoch
        avg_val_loss = np.mean(losses)
        avg_closs = np.mean(closses)
        return avg_val_loss, ious, avg_closs

    def validate_generalizability(self, epoch, plot=False):
        self.model.eval()
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, index in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
                image = x.to("cuda")
                mask = y.to("cuda")
                output = self.model(image)
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                ious = torch.cat((ious, batch_ious.flatten()))
                if plot:
                    plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
                    plt.imshow((output[0, 0].cpu().numpy() > 0.5).astype(int), alpha=0.5)
                    plt.title("IoU {} at epoch {}".format(iou(output[0, 0], mask[0, 0]), epoch))
                    plt.show()
                    plot = False  # plot one example per epoch (hacky, but works)
            return ious
