import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Models.inpainters import SegGenerator, SegDiscriminator
from DataProcessing.hyperkvasir import KvasirInpaintingDataset
from DataProcessing.etis import EtisDataset
from PipelineMods.ganlib.implementations.context_encoder.models import Generator, Discriminator
from PIL import Image
from tqdm import tqdm
from PipelineMods.polyp_inpainter import Inpainter


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def train_new_inpainter():
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    # generator = Generator(channels=3)
    # discriminator = Discriminator(channels=3)
    generator = SegGenerator()
    discriminator = SegDiscriminator()
    cuda = True
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()
    # Dataset loader
    transforms_ = [
        transforms.Resize((400, 400), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        KvasirInpaintingDataset("Datasets/HyperKvasir"),
        batch_size=8,
        shuffle=False,
        num_workers=1,
    )
    # test_dataloader = DataLoader(
    #     EtisDataset("Datasets/ETIS-LaribPolypDB"),
    #     batch_size=12,
    #     shuffle=True,
    #     num_workers=1,
    # )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

    # Initialize weights
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)
    # patch_h, patch_w = int(50 / 2 ** 3), int(50 / 2 ** 3)
    # patch = (1, patch_h, patch_w)
    # print(patch)
    for epoch in range(200):
        printed = False
        for i, (imgs, mask, masked_imgs, masked_parts, filename) in tqdm(enumerate(dataloader)):
            imgs = imgs.cuda()
            mask = mask.cuda()
            masked_imgs = masked_imgs.cuda()
            masked_parts = masked_parts.cuda()
            mask_bool = mask == 1
            # Adversarial ground truths (boxes)
            # valid = Variable(torch.Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
            # fake = Variable(torch.Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)
            valid = torch.masked_select(torch.ones_like(mask), mask_bool)
            fake = torch.masked_select(torch.zeros_like(mask), mask_bool)

            # Configure input
            imgs = Variable(imgs)
            masked_imgs = Variable(masked_imgs)
            masked_parts = Variable(masked_parts)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked_imgs)

            # Adversarial and pixelwise loss
            disc = discriminator(gen_parts)
            # print(disc)

            g_adv = adversarial_loss(torch.masked_select(disc, mask_bool), valid)
            g_pixel = pixelwise_loss(torch.masked_select(gen_parts, mask_bool), torch.masked_select(imgs, mask_bool))
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(torch.masked_select(discriminator(masked_parts), mask_bool), valid)
            fake_loss = adversarial_loss(torch.masked_select(discriminator(gen_parts.detach()), mask_bool), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()
            if not printed and epoch % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch, 400, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                )
                plt.title("Part")
                plt.imshow((gen_parts[0].detach().cpu().numpy().T))
                plt.show()
                # plt.title("Superimposed")
                # plt.imshow((gen_parts[0].detach().cpu().numpy().T))
                # plt.imshow(masked_imgs[0].detach().cpu().numpy().T)
                # plt.show()
                plt.title("Real")
                plt.imshow(masked_parts[0].detach().cpu().numpy().T)
                plt.show()
                try:
                    test = Inpainter(f"Predictors/Inpainters/deeplab-generator-{epoch}")
                    test.get_test()
                except FileNotFoundError:
                    print("Weird...")
                printed = True
        if epoch % 10 == 0:
            print("Saving")
            torch.save(generator.state_dict(), f"Predictors/Inpainters/deeplab-generator-{epoch}")
            torch.save(discriminator.state_dict(), f"Predictors/Inpainters/deeplab-discriminator-{epoch}")


if __name__ == '__main__':
    train_new_inpainter()
