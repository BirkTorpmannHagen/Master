from training.vanilla_trainer import VanillaTrainer
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
from DataProcessing.hyperkvasir import KvasirSegmentationDataset, KvasirClassificationDataset
from Models import backbones
from Models import inpainters
from Pretraining.deeplab_pretrainer import pretrain_encoder
from Tests.metrics import iou
from utils import logging
from model_of_natural_variation.model import ModelOfNaturalVariation
from losses.consistency_losses import *
import torch.nn as nn


class ConsistencyTrainer(VanillaTrainer):
    def __init__(self, model_str, id, config):
        super(ConsistencyTrainer, self).__init__(model_str, id, config)
        self.mnv = ModelOfNaturalVariation(T0=1).to(self.device)
        self.criterion = ConsistencyLoss().to(self.device)
        self.train_loader = DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir"), batch_size=8)

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            aug_img, aug_mask = self.mnv(image, mask)
            self.optimizer.zero_grad()
            output = self.model(image)
            # if np.random.rand() < 0.5:
            #     target = mask
            #     output = self.model(image)
            # else:
            #     target = aug_mask
            #     output = self.model(aug_img)
            aug_output = self.model(aug_img)
            mean_iou = torch.mean(iou(output, mask))
            loss = self.criterion(aug_mask, mask, aug_output, output, mean_iou)
            # loss = 2 * self.jaccard_test(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)

    def train(self):
        best_val_loss = 1000
        print("Starting Segmentation training")
        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            val_loss, ious = self.validate(epoch=i, plot=False)
            gen_ious = self.validate_generalizability(epoch=i, plot=False)
            logging.log_iou(f"logs/augmented-training_log_{self.model_str}_pretrainmode={self.pretrain}_{self.id}", i,
                            ious.cpu())
            mean_iou = torch.mean(ious)
            gen_iou = torch.mean(gen_ious)
            self.scheduler.step(i)
            # self.mnv.step()
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
                    f"Experiments/Data/Augmented-Pipelines/{self.model_str}/pretrainmode={self.pretrain}_{self.id}",
                    test_ious)
                print(f"Saving new best model. IID test-set mean iou: {float(np.mean(test_ious.numpy()))}")
                torch.save(self.model.state_dict(),
                           f"Predictors/Augmented/{self.model_str}/pretrainmode={self.pretrain}_{self.id}")
                print("saved in: ", f"Predictors/Augmented/{self.model_str}/pretrainmode={self.pretrain}_{self.id}")
        torch.save(self.model.state_dict(),
                   f"Predictors/Augmented/{self.model_str}/pretrainmode={self.pretrain}_{self.id}_last_epoch")

    def validate(self, epoch, plot=False):
        # todo refactor to make this prettier
        self.model.eval()
        losses = []
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, fname in self.val_loader:
                image = x.to("cuda")
                mask = y.to("cuda")
                aug_img, aug_mask = self.mnv(image, mask)
                output = self.model(image)
                aug_output = self.model(aug_img)  # todo consider train on augmented vs non-augmented?

                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                loss = self.criterion(aug_mask, mask, aug_output, output, torch.mean(batch_ious))
                losses.append(np.abs(loss.item()))
                ious = torch.cat((ious, batch_ious.cpu().flatten()))
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


class ConsistencyTrainerUsingAugmentation(ConsistencyTrainer):
    """
        Uses vanilla data augmentation with p=0.5 instead of a a custom loss
    """

    def __init__(self, model_str, id, config):
        super(ConsistencyTrainerUsingAugmentation, self).__init__(model_str, id, config)
        self.jaccard_test = vanilla_losses.JaccardLoss()

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            aug_img, aug_mask = self.mnv(image, mask)
            self.optimizer.zero_grad()
            if np.random.rand() < 0.5:
                target = mask
                output = self.model(image)
            else:
                target = aug_mask
                output = self.model(aug_img)
            aug_output = self.model(aug_img)
            mean_iou = torch.mean(iou(output, mask))
            loss = self.jaccard_test(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)


class AdversarialConsistencyTrainer(ConsistencyTrainer):
    """
        Adversariall samples difficult
    """

    def __init__(self, model_str, id, config):
        super(ConsistencyTrainer, self).__init__(model_str, id, config)
        self.mnv = ModelOfNaturalVariation(T0=0.25).to(self.device)
        self.num_adv_samples = 25
        self.naked_closs = NakedConsistencyLoss()
        self.criterion = ConsistencyLoss().to(self.device)

    def sample_adversarial(self, image, mask, output):
        self.model.eval()
        aug_img, aug_mask = None, None  #
        max_severity = -10
        with torch.no_grad():
            for i in range(self.num_adv_samples):
                adv_aug_img, adv_aug_mask = self.mnv(image, mask)
                adv_aug_output = self.model(adv_aug_img)
                severity = self.naked_closs(adv_aug_mask, mask, adv_aug_output, output)
                if severity > max_severity:
                    max_severity = severity
                    aug_img = adv_aug_img
                    aug_mask = adv_aug_mask
        self.model.train()
        return aug_img, aug_mask

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            output = self.model(image)
            aug_img, aug_mask = self.sample_adversarial(image, mask, output)
            self.optimizer.zero_grad()
            mean_iou = torch.mean(iou(output, mask))
            aug_output = self.model(aug_img)
            loss = self.criterion(aug_mask, mask, aug_output, output, mean_iou)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        self.mnv.step()  # increase difficulty
        return np.mean(losses)