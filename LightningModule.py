import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from models.LBAMModel import LBAMModel, VGG16FeatureExtractor
import torch.optim as optim
import torch
from loss.InpaintingLossLightning import InpaintingLossWithGAN
import math
import pytorch_ssim
from pathlib import Path
from torchvision.utils import save_image


class LBAMModule(pl.LightningModule):
    def __init__(self, configs: dict):
        super().__init__()
        self.LBAM_model = LBAMModel(4, 3)
        if configs['pretrained_torch'] is not None:
            self.LBAM_model.load_state_dict(torch.load(configs['pretrained_torch']))
        self.save_hyperparameters()
        self.loss = InpaintingLossWithGAN(VGG16FeatureExtractor(), lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)
        self.image_directory = configs['image_directory']

    def training_step(self, batch, batch_idx):
        input_images, ground_truth, masks = batch
        prediction = self.LBAM_model(input_images, masks)
        loss, d_loss, hole_loss, valid_area_loss, prc_loss, style_loss, g_loss = self.loss(input_images[:, 0:3, :, :],
                                                                                           masks, prediction,
                                                                                           ground_truth)
        self.log('train_loss', loss)
        self.log('discriminator_loss', d_loss)
        self.log('hole_loss', hole_loss)
        self.log('train_loss', loss)
        self.log('valid_loss', valid_area_loss)
        self.log('perceptual_loss', prc_loss)
        self.log('style_loss', style_loss)
        self.log('train_loss', loss)
        self.log('joint_loss', g_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_images, ground_truth, masks = batch
        prediction = self.LBAM_model(input_images, masks)
        damaged = ground_truth * masks
        generated_image = ground_truth * masks + prediction * (1 - masks)
        batch_mse = ((ground_truth - generated_image) ** 2).mean()
        psnr = 10 * math.log10(1 / batch_mse)
        ssim = pytorch_ssim.ssim(ground_truth * 255, generated_image * 255)
        l1_loss = nn.L1Loss()(generated_image, ground_truth)
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim)
        self.log('val_l1_loss', l1_loss)
        outputs = torch.Tensor(4 * ground_truth.size()[0], ground_truth.size()[1], 256, 256)
        for i in range(ground_truth.size()[0]):
            outputs[4 * i] = masks[i]
            outputs[4 * i + 1] = damaged[i]
            outputs[4 * i + 2] = generated_image[i]
            outputs[4 * i + 3] = ground_truth[i]
        image_destination = Path(self.image_directory, f'results-{batch_idx}.png')
        save_image(outputs, image_destination)
        self.logger.log_image(key='validation_images', images=[image_destination])

    def configure_optimizers(self):
        return optim.Adam(self.LBAM_model.parameters(), lr=0.0001, betas=(0.5, 0.9))
