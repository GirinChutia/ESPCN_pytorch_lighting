import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from pl_modules.litmodules import ESPCNLitModule
from pl_modules.datamodules import BDSDataModule
import time,os
timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

@dataclass
class TrainingConfig:
    data_dir: str = 'D:/Work/learning/paper_implementation/super_resolution/Dataset/BSR_bsds500/BSR/BSDS500/data/images'
    log_dir: str = "logs"
    checkpoint_dir: str = "pl_checkpoints"
    
    upscale_factor: int = 3
    epoch: int = 100
    num_workers: int = 4
    device:str = 'gpu'
    
    use_mlflow: bool = True
    mlflow_log_dir: str = "logs/mlruns"
    mlflow_experiment_name: str = "exp1"
    mlflow_run_name: str = "run1"
    
@dataclass
class ModelHyperParameters:
    batch_size: int = 64
    learning_rate: float = 0.01
    normalize_input: bool = True

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, "train_config")
    parser.add_arguments(ModelHyperParameters, "hparams")
    args = parser.parse_args()
    
    hparams: ModelHyperParameters = args.hparams
    train_config: TrainingConfig = args.train_config
    
    batch_size = hparams.batch_size
    learning_rate = hparams.learning_rate
    normalize_input = hparams.normalize_input
    
    upscale_factor = train_config.upscale_factor
    epoch = train_config.epoch
    num_workers = train_config.num_workers
    default_root_dir = os.path.join(train_config.checkpoint_dir,timestr)
    data_dir=train_config.data_dir
    
    # RGB -----------
    # means = [0.4491, 0.4482, 0.3698]
    # stds = [0.2422, 0.2272, 0.2335]
    
    # Y Channels only -----
    means = 0.4412
    stds = 0.2194

    bsd_dm = BDSDataModule(image_dir=data_dir,
                           upscale_factor=upscale_factor,
                           batch_size=batch_size,
                           num_workers = num_workers,
                           crop_size=256,
                           normalize=normalize_input,
                           means=means,
                           stds=stds)
    
    model = ESPCNLitModule(upscale_factor=upscale_factor,
                           lr=learning_rate,
                           channels=1)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_psnr',
                                          mode='max',
                                          save_last=True,
                                          verbose=False,
                                          dirpath=default_root_dir,
                                          filename='best-model-{epoch:02d}-{val_psnr:.2f}')
    
    csv_log_dir = os.path.join(train_config.log_dir,timestr)
    csvlogger = CSVLogger(csv_log_dir, name="csvlogs")
    mlf_logger = MLFlowLogger(experiment_name=train_config.mlflow_experiment_name,
                              run_name=train_config.mlflow_run_name,
                              tracking_uri=train_config.mlflow_log_dir)
    
    mlf_logger.log_hyperparams(vars(hparams))
    mlf_logger.log_hyperparams(vars(train_config))
    mlf_logger.log_hyperparams({'csv_log_dir':csv_log_dir,'checkpoint_log_dir':default_root_dir})
    csvlogger.log_hyperparams(args)
    csvlogger.log_hyperparams({'csv_log_dir':csv_log_dir,'checkpoint_log_dir':default_root_dir})
    
    trainer = pl.Trainer(accelerator=train_config.device,
                         devices=1,
                         min_epochs=1,
                         max_epochs=epoch,
                         log_every_n_steps=4,
                         check_val_every_n_epoch=1, 
                         default_root_dir=default_root_dir,
                         callbacks=[checkpoint_callback],
                         logger=[mlf_logger,csvlogger])
    
    trainer.fit(model,
                datamodule=bsd_dm)
