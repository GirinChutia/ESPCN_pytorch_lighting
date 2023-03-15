import pytorch_lightning as pl
import torch.utils.data as data
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
from math import log10
import torch.optim as optim
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import CSVLogger

class SRNet(pl.LightningModule):
    def __init__(self, upscale_factor,lr=0.01):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        self._initialize_weights()
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
        
    def training_step(self, batch, batch_idx):
        input, target = batch
        y_hat = self(input)    
        loss = self.criterion(y_hat, target)
        psnr = 10 * log10(1 / loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss':loss}  # optimization is done on the loss key  
    
    def validation_step(self, batch, batch_idx):
        input, target = batch
        y_hat = self(input)    
        loss = self.criterion(y_hat, target)
        psnr = 10 * log10(1 / loss.item())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True, logger=True)  # MLFLOW is not logging validation logs
        return {'val_loss':loss,'val_psnr':loss} 
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    