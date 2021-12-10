from Models.backbones import *
from Tests.metrics import iou
from DataProcessing.hyperkvasir import KvasirSegmentationDataset
from DataProcessing.etis import EtisDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def eval(dataset, model):
    model.eval()
    ious = torch.empty((0,))
    with torch.no_grad():
        for x, y, fname in dataset:
            image = x.to("cuda")
            mask = y.to("cuda")
            output = model(image)
            batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
            ious = torch.cat((ious, batch_ious.flatten()))
    return ious.numpy()


def get_generalizability_gap(modelpath):
    if modelpath.split("/")[1] == "DeepLab":
        model = DeepLab("imagenet").to("cuda")
        model.load_state_dict(torch.load(modelpath))
    else:
        raise NotImplementedError(modelpath.split("/")[1])

    kvasir = DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))
    etis = DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB"))
    
    kvasir_ious = eval(kvasir, model)
    etis_ious = eval(etis, model)
    print(np.mean(kvasir_ious),
          np.mean(etis_ious))


if __name__ == '__main__':
    get_generalizability_gap("Predictors/DeepLab/imagenet_pretrain/pretrainmode=imagenet_0")
