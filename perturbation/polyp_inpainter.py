import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

# from perturbation.gmcnn.model.net_with_dropout import InpaintingModel_GMCNN_Given_Mask
from data.hyperkvasir import *
from models.inpainters import SegGenerator
from perturbation.perturbator import RandomDraw


class Inpainter:
    # wrapper around gmcnn
    def __init__(self, path_to_state_dict):
        super(Inpainter, self).__init__()
        self.config = None
        self.model = SegGenerator()
        self.model.load_state_dict(torch.load(path_to_state_dict))
        self.model.eval()
        self.perturbator = RandomDraw()

    def forward(self, img, mask, masked_image=None):
        mask = mask.unsqueeze(1)
        masked_image = img * (1 - mask)
        polyp = self.model(masked_image)
        merged = (1 - mask) * img + (polyp * mask)
        return merged, polyp

    def add_polyp(self, img, old_mask):
        new_polyp_mask = np.expand_dims(self.perturbator(rad=0.25), -1)
        total_mask = np.clip(new_polyp_mask[0] + old_mask, 0, 1)
        masked = img * (1 - new_polyp_mask)
        with torch.no_grad():
            polyp = self.model(torch.Tensor(masked).permute(-1, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
        inpainted = (1 - new_polyp_mask) * img + (polyp * new_polyp_mask)
        return inpainted.astype(np.float32), total_mask.astype(np.float32)

    def get_test(self, split="test"):
        for i, (image, mask, masked_image, part, fname) in enumerate(
                DataLoader(KvasirSyntheticDataset("Datasets/HyperKvasir", split="test"),
                           batch_size=4)):
            merged, mask = self.add_polyp(image, mask)
            plt.title("Inpainted image")
            plt.imshow(merged[0].T)
            plt.show()
            # plt.savefig(f"perturbation/inpaint_examples/{i}")


if __name__ == '__main__':
    inpainter = Inpainter("Predictors/Inpainters/no-pretrain-deeplab-generator-4990")
    inpainter.get_test()
