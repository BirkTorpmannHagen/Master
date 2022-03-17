import torch
import torch.nn as nn


def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    outputs = (outputs.squeeze(1) > 0.5).int()
    labels = (labels > 0.5).int()
    intersection = torch.sum((outputs & labels).float())
    union = torch.sum((outputs | labels).float())

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou


class SegmentationInconsistencyScore(nn.Module):  # todo refactor module requirement
    """
        Implementation of SIS
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
