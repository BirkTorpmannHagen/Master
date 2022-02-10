from os import listdir

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.logging import log_full
from DataProcessing.etis import EtisDataset
from DataProcessing.hyperkvasir import KvasirSegmentationDataset
from DataProcessing.endocv import EndoCV2020
from DataProcessing.cvc import CVC_ClinicDB
from Models.segmentation_models import *
from Tests.metrics import iou


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
    model = DeepLab().to("cuda")
    model.load_state_dict(torch.load(modelpath))
    # if modelpath.split("/")[1] == "DeepLab":
    #     model = DeepLab("imagenet").to("cuda")
    #     model.load_state_dict(torch.load(modelpath))
    # else:
    #     raise NotImplementedError(modelpath.split("/")[1])

    kvasir = DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))
    etis = DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB"))
    EndoCV = DataLoader(EndoCV2020("Datasets/EndoCV2020-Endoscopy-Disease-Detection-Segmentation-subChallenge_data"))
    CVC = DataLoader(CVC_ClinicDB("Datasets/CVC-ClinicDB"))
    endocv_ious = eval(EndoCV, model)
    kvasir_ious = eval(kvasir, model)
    etis_ious = eval(etis, model)
    cvc_ious = eval(CVC, model)
    print(
        f"{modelpath} \t \t {np.mean(kvasir_ious):.4f} \t {np.mean(etis_ious):.4f} \t {np.mean(endocv_ious):.4f} \t {np.mean(cvc_ious):.4f}")
    return kvasir_ious, etis_ious


if __name__ == '__main__':
    # get_generalizability_gap("Predictors/DeepLab/imagenet_pretrain/pretrainmode=imagenet_0")
    # get_generalizability_gap("Predictors/DeepLab/imagenet_pretrain/pretrainmode=imagenet_1")
    # get_generalizability_gap("Predictors/DeepLab/imagenet_pretrain/pretrainmode=imagenet_2")
    # get_generalizability_gap("Predictors/DeepLab/imagenet_pretrain/pretrainmode=imagenet_3")
    # get_generalizability_gap("Predictors/Augmented/DeepLab-pretrainmode=imagenet_-10")
    # get_generalizability_gap("Predictors/Augmented/DeepLab-pretrainmode=imagenet_-9")
    # get_generalizability_gap("Predictors/Augmented/DeepLab/pretrainmode=imagenet_0")
    # get_generalizability_gap("Predictors/Augmented/DeepLab/pretrainmode=imagenet_1")
    get_generalizability_gap("Predictors/Vanilla/DeepLab/1_last_epoch")
    get_generalizability_gap("Predictors/Augmented/DeepLab/resolution_test_last_epoch")
    get_generalizability_gap("Predictors/Augmented/DeepLab/1_last_epoch")

    # for fname in sorted(listdir("Predictors/Augmented/DeepLab/")):
    #     try:
    #         get_generalizability_gap(f"Predictors/Augmented/DeepLab/{fname}")
    #     except Exception as e:
    #         print(e)
    #         continue
    # print("vanilla")
    # for fname in listdir("Predictors/Vanilla/DeepLab/"):
    #     try:
    #         get_generalizability_gap(f"Predictors/Vanilla/DeepLab/{fname}")
    #     except:
    #         continue
    # get_generalizability_gap("Predictors/DeepLab/pretrainmode=imagenet_1000_epochs_2")
    # get_generalizability_gap("Predictors/Augmented/DeepLab-pretrainmode=imagenet_test2")

# %%
