import matplotlib.pyplot as plt
import numpy as np
import torch.optim.optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.hyperkvasir import KvasirSegmentationDataset, KvasirMNVset
from evaluation.metrics import iou
from losses.consistency_losses import *
from perturbation.model import ModelOfNaturalVariation
from training.vanilla_trainer import VanillaTrainer
from utils import logging
from models.ensembles import TrainedEnsemble


class ConsistencyTrainer(VanillaTrainer):
    def __init__(self, id, config):
        super(ConsistencyTrainer, self).__init__(id, config)
        self.criterion = ConsistencyLoss().to(self.device)
        self.nakedcloss = NakedConsistencyLoss()

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            aug_img, aug_mask = self.mnv(image, mask)
            self.optimizer.zero_grad()
            output = self.model(image)
            aug_output = self.model(aug_img)
            mean_iou = torch.mean(iou(output, mask))
            loss = self.criterion(aug_mask, mask, aug_output, output, mean_iou)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)

    def train(self):

        best_val_loss = 1000
        best_consistency = 0
        print("Starting Segmentation training")
        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            val_loss, ious, closs = self.validate(epoch=i, plot=False)
            gen_ious = self.validate_generalizability(epoch=i, plot=False)
            mean_iou = float(torch.mean(ious))
            gen_iou = float(torch.mean(gen_ious))
            consistency = 1 - np.mean(closs)
            test_iou = np.mean(self.test().numpy())

            self.config["lr"] = [group['lr'] for group in self.optimizer.param_groups]
            logging.log_full(epoch=i, id=self.id, config=self.config, result_dict=
            {"train_loss": training_loss, "val_loss": val_loss,
             "iid_val_iou": mean_iou, "iid_test_iou": test_iou, "ood_iou": gen_iou,
             "consistency": consistency}, type="consistency")

            self.scheduler.step(i)
            # self.mnv.step()
            print(
                f"Epoch {i} of {self.epochs} \t"
                f" lr={[group['lr'] for group in self.optimizer.param_groups]} \t"
                f" loss={training_loss} \t"
                f" val_loss={val_loss} \t"
                f" ood_iou={gen_iou}\t"
                f" val_iou={mean_iou} \t"
                f" gen_prop={gen_iou / mean_iou} \t,"
                f" consistency={consistency}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                np.save(
                    f"experiments/Data/Augmented-Pipelines/{self.model_str}/{self.id}",
                    test_iou)
                print(f"Saving new best model. IID test-set mean iou: {test_iou}")
                torch.save(self.model.state_dict(),
                           f"Predictors/Augmented/{self.model_str}/{self.id}")
                print("saved in: ", f"Predictors/Augmented/{self.model_str}/{self.id}")

            if consistency > best_consistency:
                best_consistency = consistency
                torch.save(self.model.state_dict(),
                           f"Predictors/Augmented/{self.model_str}/maximum_consistency{self.id}")
        torch.save(self.model.state_dict(),
                   f"Predictors/Augmented/{self.model_str}/{self.id}_last_epoch")

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

    def __init__(self, id, config):
        super(ConsistencyTrainerUsingAugmentation, self).__init__(id, config)
        self.jaccard = vanilla_losses.JaccardLoss()
        self.prob = 0
        self.dataset = KvasirMNVset("Datasets/HyperKvasir", "train", inpaint=config["use_inpainter"])
        self.train_loader = DataLoader(self.dataset, batch_size=config["batch_size"], shuffle=True)

    def get_iou_weights(self, image, mask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        return torch.mean(iou(output, mask))

    def get_consistency(self, image, mask, augmented, augmask):
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        self.model.train()
        return torch.mean(self.nakedcloss(output, mask, augmented, augmask))

    def train_epoch(self):
        self.model.train()
        losses = []
        plotted = False
        for x, y, fname, flag in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.jaccard(output, mask)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)


class AdversarialConsistencyTrainer(ConsistencyTrainer):
    """
        Adversariall samples difficult
    """

    def __init__(self, id, config):
        super(ConsistencyTrainer, self).__init__(id, config)
        self.mnv = ModelOfNaturalVariation(T0=1).to(self.device)
        self.num_adv_samples = 10
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
            self.optimizer.zero_grad()
            output = self.model(image)
            aug_img, aug_mask = self.sample_adversarial(image, mask, output)
            # aug_img, aug_mask = self.mnv(image, mask)
            aug_output = self.model(aug_img)
            mean_iou = torch.mean(iou(output, mask))
            loss = self.criterion(aug_mask, mask, aug_output, output, mean_iou)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)


class StrictConsistencyTrainer(ConsistencyTrainer):
    def __init__(self, id, config):
        super(StrictConsistencyTrainer, self).__init__(id, config)
        self.criterion = StrictConsistencyLoss()


class ConsistencyTrainerUsingControlledAugmentation(ConsistencyTrainer):
    """
        Uses vanilla data augmentation with p=0.5 instead of a a custom loss and has two samples
    """

    def __init__(self, id, config):
        super(ConsistencyTrainerUsingControlledAugmentation, self).__init__(id, config)
        self.jaccard = vanilla_losses.JaccardLoss()
        self.mnv = ModelOfNaturalVariation(1)

    def train_epoch(self):
        self.model.train()
        losses = []
        plotted = False
        for x, y, fname in self.train_loader:
            image = x.to("cuda")
            mask = y.to("cuda")
            aug_img, aug_mask = self.mnv(image, mask)
            img_batch = torch.cat((image, aug_img))
            mask_batch = torch.cat((mask, aug_mask))
            self.optimizer.zero_grad()
            output = self.model(img_batch)
            loss = self.jaccard(output, mask_batch)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)


class EnsembleConsistencyTrainer(ConsistencyTrainer):
    def __init__(self, id, config):
        super(EnsembleConsistencyTrainer, self).__init__(id, config)
        self.model = TrainedEnsemble("Singular")
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
