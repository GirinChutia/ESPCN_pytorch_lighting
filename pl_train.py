import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules import BDS500DataModule
from modelmodules import SRNet
from dataclasses import dataclass
from simple_parsing import ArgumentParser

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

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, "train_config")
    parser.add_arguments(ModelHyperParameters, "hparams")
    args = parser.parse_args()
    
    hparams: ModelHyperParameters = args.hparams
    train_config: TrainingConfig = args.train_config
    
    batch_size = hparams.batch_size
    learning_rate = hparams.learning_rate
    
    upscale_factor = train_config.upscale_factor
    epoch = train_config.epoch
    num_workers = train_config.num_workers
    default_root_dir=train_config.checkpoint_dir
    data_dir=train_config.data_dir
    
    bsd_dm = BDS500DataModule(image_dir=data_dir,
                              upscale_factor=upscale_factor,
                              batch_size=batch_size,
                              num_workers = num_workers)
    
    model = SRNet(upscale_factor=upscale_factor,
                  lr=learning_rate)
    
    checkpoint_callback = ModelCheckpoint(monitor='val_psnr',
                                          mode='max',
                                          save_last=True,
                                          verbose=True,
                                          dirpath=default_root_dir,
                                          filename='best-model-{epoch:02d}-{val_psnr:.2f}')
    
    csvlogger = CSVLogger(train_config.log_dir, name="csvlogs")
    mlf_logger = MLFlowLogger(experiment_name=train_config.mlflow_experiment_name,
                              run_name=train_config.mlflow_run_name,
                              tracking_uri=train_config.mlflow_log_dir)
    
    mlf_logger.log_hyperparams(vars(hparams))
    mlf_logger.log_hyperparams(vars(train_config))
    csvlogger.log_hyperparams(args)
    
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
