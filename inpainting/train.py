import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import torch
import torch.nn as nn
from data.data import InpaintingDataset, ToTensor
from model.net import InpaintingModel_GMCNN
from options.train_options import TrainOptions
from util.utils import getLatest
from data.hyperkvasir import KvasirSegmentationDataset
import torch.nn.functional as F
from inpainting.model.basemodel import BaseModel
from inpainting.model.basenet import BaseNet
from inpainting.model.loss import WGANLoss, IDMRFLoss
from inpainting.model.layer import init_weights, PureUpsampling, ConfidenceDrivenMaskLayer, SpectralNorm
from inpainting.model.net import *

config = {"model": "DeepLab",
          "device": "cuda",
          "lr": 0.00001,
          "batch_size": 8,
          "epochs": 250}
DATASET_PATH = ""


class gmcnn_inpainter_trainer:
    def __init__(self, config):
        self.model = InpaintingModel_GMCNN(in_channels=3, opt=config)
        self.dataset = KvasirSegmentationDataset(DATASET_PATH, augment=True)  # todo make inpainting dataset
        self.dataloader = DataLoader(self.dataset, batch_size=config["batch_size"], shuffle=True)
        self.device = config["device"]
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()
        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()

        self.netGM = GMCNN(3, out_channels=3, cnum=32, act=F.elu, norm=F.instance_norm).cuda()
        init_weights(self.netGM)
        self.model_names = ['GM']

        self.netD = None

        self.optimizer_G = torch.optim.Adam(self.netGM.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.optimizer_D = None

        self.wganloss = None
        self.recloss = nn.L1Loss()
        self.aeloss = nn.L1Loss()
        self.mrfloss = None
        # self.lambda_adv = opt.lambda_adv
        # self.lambda_rec = opt.lambda_rec
        # self.lambda_ae = opt.lambda_ae
        # self.lambda_gp = opt.lambda_gp
        # self.lambda_mrf = opt.lambda_mrf
        self.G_loss = None
        self.G_loss_reconstruction = None
        self.G_loss_mrf = None
        self.G_loss_adv, self.G_loss_adv_local = None, None
        self.G_loss_ae = None
        self.D_loss, self.D_loss_local = None, None
        self.GAN_loss = None

        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None

        self.completed, self.completed_local = None, None
        self.completed_logit, self.completed_local_logit = None, None
        self.gt_logit, self.gt_local_logit = None, None

        self.pred = None
        self.netD = GlobalLocalDiscriminator(3, cnum=64, act=F.elu,
                                             spectral_norm=True,
                                             g_fc_channels=512 // 16 * 512 // 16 * 64 * 4,
                                             l_fc_channels=512 // 16 * 512 // 16 * 64 * 4).to(self.device)
        init_weights(self.netD)
        self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=config["lr"],
                                            betas=(0.5, 0.9))
        self.wganloss = WGANLoss()
        self.mrfloss = IDMRFLoss()

    def train(self):
        for epoch in range(config["epochs"]):
            self.train_epoch(epoch)
            ret_loss = self.model.get_current_losses()
            self.model.save_networks(epoch + 1)

    def train_epoch(self, epoch):
        for img, mask, fname in self.dataloader:
            img, mask = img.to(self.device), mask.to(self.device)
            img_in = img * (1 - mask)
            self.gen_in = torch.cat((img_in, mask), 1)
            self.model.setInput(img, mask)
            self.model.optimize_parameters()

            self.pred = self.netGM(self.gin)
            self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
            self.completed_local = self.completed

            for i in range(5):  # train discriminator 5 times interleaved
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.forward_D()
                self.backward_D()
                self.optimizer_D.step()

            self.optimizer_G.zero_grad()
            self.forward_G()
            self.backward_G()
            self.optimizer_G.step()
            # TODO come back here to finish gmcnn

    def forward_G(self):
        self.G_loss_reconstruction = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_reconstruction = self.G_loss_reconstruction / torch.mean(self.mask_01)
        self.G_loss_ae = self.aeloss(self.pred * (1 - self.mask_01), self.gt.detach() * (1 - self.mask_01))
        self.G_loss_ae = self.G_loss_ae / torch.mean(1 - self.mask_01)
        self.G_loss = self.lambda_rec * self.G_loss_reconstruction + self.lambda_ae * self.G_loss_ae

        self.completed_logit, self.completed_local_logit = self.netD(self.completed, self.completed_local)
        self.G_loss_mrf = self.mrfloss((self.completed_local + 1) / 2.0, (self.gt_local.detach() + 1) / 2.0)
        self.G_loss = self.G_loss + self.lambda_mrf * self.G_loss_mrf

        self.G_loss_adv = -self.completed_logit.mean()
        self.G_loss_adv_local = -self.completed_local_logit.mean()
        self.G_loss = self.G_loss + self.lambda_adv * (self.G_loss_adv + self.G_loss_adv_local)

    def forward_D(self):
        self.completed_logit, self.completed_local_logit = self.netD(self.completed.detach(),
                                                                     self.completed_local.detach())
        self.gt_logit, self.gt_local_logit = self.netD(self.gt, self.gt_local)
        # hinge loss
        self.D_loss_local = nn.ReLU()(1.0 - self.gt_local_logit).mean() + nn.ReLU()(
            1.0 + self.completed_local_logit).mean()
        self.D_loss = nn.ReLU()(1.0 - self.gt_logit).mean() + nn.ReLU()(1.0 + self.completed_logit).mean()
        self.D_loss = self.D_loss + self.D_loss_local

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)
