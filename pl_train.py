import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import CSVLogger
from datamodules import BDS500DataModule
from modelmodules import SRNet

upscale_factor = 3
epoch = 100
batch_size = 64
num_workers = 2
learning_rate = 0.01

bsd_dm = BDS500DataModule(image_dir=r'D:\Work\learning\paper_implementation\super_resolution\Dataset\BSR_bsds500\BSR\BSDS500\data\images',
                            upscale_factor=upscale_factor,
                            batch_size=batch_size,
                            num_workers = num_workers)

model = SRNet(upscale_factor=upscale_factor,
                lr=learning_rate)

mlf_logger = MLFlowLogger(experiment_name="lightning_logs",
                          run_name='run1',
                          tracking_uri="./mlruns")
    
if __name__ == "__main__":
    
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         min_epochs=1,
                         max_epochs=epoch,
                         log_every_n_steps=4,
                         check_val_every_n_epoch=1, 
                         logger=[mlf_logger])
    
    trainer.fit(model,
                datamodule=bsd_dm)
