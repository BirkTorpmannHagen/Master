from training.vanilla_trainer import VanillaTrainer
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch.optim.optimizer
from torch.utils.data import DataLoader
import DataProcessing.augmentation as aug
from DataProcessing.hyperkvasir import KvasirSegmentationDataset, KvasirClassificationDataset
from DataProcessing.etis import EtisDataset
from Models import backbones
from Models import inpainters
from Pretraining.deeplab_pretrainer import pretrain_encoder
from Tests.metrics import iou
from utils import logging
from model_of_natural_variation.model import ModelOfNaturalVariation
import torch.nn as nn


class PerturbationLoss(nn.Module):
    def __init__(self):
        super(PerturbationLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        return torch.sum((difference(difference(new_mask, old_mask),
                                     difference(new_seg, old_seg))
                          ) / \
                         torch.sum(torch.clamp(new_seg + new_mask + old_mask + old_seg, 0,
                                               1))) + self.epsilon  # normalizing factor

        # newly predicted pixels that should not be predicted
        # term1 = difference(new_seg, old_seg) * new_seg * (1 - new_mask)
        # # changes to the correct segmentation that also is not in the new segmentation
        # term2 = (1 - new_seg) * old_mask * old_seg * (1 - new_mask)
        # # normalizing factor
        # divisor = torch.sum(torch.clamp(new_mask + old_mask, min=0, max=1))
        # return (term1 + term2 + self.epsilon) / (divisor + self.epsilon)


class ContrastiveTrainer(VanillaTrainer):
    def __init__(self, model_str, id, config):
        super(ContrastiveTrainer, self).__init__(model_str, id, config)
        self.mnv = ModelOfNaturalVariation().to(self.device)
        self.contrastive_criterion = PerturbationLoss().to(self.device)
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
            aug_output = self.model(aug_img)  # todo consider train on augmented vs non-augmented?
            loss_van = self.criterion(output, mask)
            loss_cont = self.contrastive_criterion(aug_mask, mask, aug_output, output)
            # loss_cont = 0
            loss = loss_cont + loss_van
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
            gen_losses, gen_ious = self.validate_generalizabiliy(epoch=i, plot=True)
            logging.log_iou(f"logs/augmented-training_log_{self.model_str}_pretrainmode={self.pretrain}_{self.id}", i,
                            ious)
            mean_iou = torch.mean(ious)
            gen_iou = torch.mean(gen_ious)
            self.scheduler.step(i)
            self.mnv.step()
            print(
                f"Epoch {i} of {self.epochs} \t lr={[group['lr'] for group in self.optimizer.param_groups]}, loss={training_loss} \t val_loss={validation_loss} \t generalizability={gen_iou} mean_iou={mean_iou}, prop={gen_iou / mean_iou}")

            if mean_iou > best_iou:
                best_iou = mean_iou
                np.save(
                    f"Experiments/Data/Augmented-Pipelines/{self.model_str}/pretrainmode={self.pretrain}_{self.id}",
                    test_ious)

                print(f"Saving new best model. Test-set mean iou: {float(np.mean(test_ious.numpy()))}")
                torch.save(self.model.state_dict(),
                           f"Predictors/Augmented/{self.model_str}-pretrainmode={self.pretrain}_{self.id}")

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

    def validate_generalizabiliy(self, epoch, plot=False):
        self.model.eval()
        losses = []
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, index in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
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
