import torch.nn as nn
import torch

class perturbation_loss(nn.Module):
    def __init__(self):
        super(perturbation_loss, self).__init__()
    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1*(1-mask2)+mask2*(1-mask1)
        #newly predicted pixels that should not be predicted
        term1 = difference(new_seg, old_seg)*new_seg*(1-new_mask)
        #changes to the correct segmentation that also is not in the new segmentation
        term2 = (1-new_seg)*old_mask*old_seg*(1-new_mask)
        #normalizing factor
        divisor = torch.sum(torch.clamp(new_mask+old_mask, min=0, max=1))
        return (term1+term2)/divisor

class ContrastiveInpaintingTrainer():
    def __init__(self):
        loss = perturbation_loss()

    def train_epoch(self):
        pass

    def train(self):
        pass

if __name__ == '__main__':
    model
    loss = perturbation_loss()
