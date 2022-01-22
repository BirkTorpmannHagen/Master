import segmentation_models_pytorch.utils.losses as vanilla_losses
import torch
import torch.nn as nn


class ConsistencyLoss(nn.Module):
    """

    """

    def __init__(self, adaptive=True):
        super(ConsistencyLoss, self).__init__()
        self.epsilon = 1e-5
        self.adaptive = adaptive
        self.jaccard = vanilla_losses.JaccardLoss()

    def forward(self, new_mask, old_mask, new_seg, old_seg, iou_weight=None):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        vanilla_jaccard = vanilla_losses.JaccardLoss()(old_seg, old_mask)

        perturbation_loss = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)

        # ~bd + ~ac + b~d + a~c
        # perturbation_loss = (1 - new_mask) * new_seg + (1 - old_mask) * old_seg + new_mask * (
        #         1 - new_seg) + old_mask * (1 - old_seg)
        if self.adaptive:
            return (1 - iou_weight) * vanilla_jaccard + iou_weight * perturbation_loss
        return 0.5 * vanilla_jaccard + 0.5 * perturbation_loss
        # return perturbation_loss
        # return self.jaccard(old_seg, old_mask) + self.jaccard(new_seg, new_mask)


class NakedConsistencyLoss(nn.Module):
    """

    """

    def __init__(self):
        super(NakedConsistencyLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return mask1 * (1 - mask2) + mask2 * (1 - mask1)

        perturbation_loss = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)  # normalizing factor
        # perturbation_loss = torch.sum(perturbation_loss)
        return perturbation_loss
