import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.init as init
from math import log10
import torch.optim as optim
import torch
from .model import ESPCN_model

class ESPCNLitModule(pl.LightningModule):
    
    def __init__(self, upscale_factor=3,lr=0.01,channels=1):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.upscale_factor = upscale_factor
        self.lr = lr
        self.ESPCN_model = ESPCN_model(scale=self.upscale_factor,
                                       channels=channels)
    def forward(self, x):
        x = self.ESPCN_model(x)
        return x
        
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler' : torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                           milestones=[35,80],
                                                                           gamma=0.1,
                                                                           verbose=True),
                        'name': 'MultiSetLR_Scheduler'}
        return [optimizer], [lr_scheduler]