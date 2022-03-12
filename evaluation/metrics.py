import torch
import torch.nn as nn


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = (outputs.squeeze(1) > 0.5).int()  # BATCH x 1 x H x W => BATCH x H x W
    labels = (labels > 0.5).int()
    intersection = torch.sum((outputs & labels).float())  # Will be zero if Truth=0 or Prediction=0
    union = torch.sum((outputs | labels).float())  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou  # Or thresholded.mean() if you are interested in average across the batch


class SegmentationInconsistencyScore(nn.Module):
    """

    """

    def __init__(self):
        super(SegmentationInconsistencyScore, self).__init__()
        self.epsilon = 1e-5

    def forward(self, new_mask, old_mask, new_seg, old_seg):
        def difference(mask1, mask2):
            return torch.round(mask1) * (1 - torch.round(mask2)) + torch.round(mask2) * (
                    1 - torch.round(mask1))

        perturbation_loss = torch.sum(
            difference(
                difference(new_mask, old_mask),
                difference(new_seg, old_seg))
        ) / torch.sum(torch.clamp(new_mask + old_mask + new_seg + old_seg, 0, 1) + self.epsilon)  # normalizing factor
        return perturbation_loss


if __name__ == '__main__':
    test = torch.zeros(10, 10)
    test[:3, :3] = 1
    test2 = torch.zeros(10, 10)
    test2[:3, :3] = 1
    print(iou(test, test2))
