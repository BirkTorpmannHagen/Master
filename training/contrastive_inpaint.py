from Models import backbones, inpainters
import torch.nn as nn
import torch
from training.vanilla_trainer import VanillaTrainer
from model_of_natural_variation import polyp_inpainter, perturbator


class perturbation_loss(nn.Module):
    def __init__(self):
        super(perturbation_loss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        # newly predicted pixels that should not be predicted
        term1 = difference(new_seg, old_seg) * new_seg * (1 - new_mask)
        # changes to the correct segmentation that also is not in the new segmentation
        term2 = (1 - new_seg) * old_mask * old_seg * (1 - new_mask)
        # normalizing factor
        divisor = torch.sum(torch.clamp(new_mask + old_mask, min=0, max=1))
        return (term1 + term2 + self.epsilon) / (divisor + self.epsilon)


class ContrastiveInpaintingTrainer(VanillaTrainer):
    def __init__(self, model_str, id, config):
        super(ContrastiveInpaintingTrainer, self).__init__(model_str, id, config)
        self.challange_loss = perturbation_loss()
        self.perturbator = perturbator.RandomDraw()

    def train_epoch(self):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    model = backbones.DeepLab()
    inpainter = inpainters.SegGenerator()
    inpainter.load_state_dict(torch.load(""))
    loss = perturbation_loss()
