import torch.utils.data as data
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from torch.utils.data import DataLoader
from PIL import Image
import pytorch_lightning as pl

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in \
            os.listdir(image_dir) if DatasetFromFolder.is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = DatasetFromFolder.load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)
    
    @staticmethod
    def load_img(filepath):
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
        return y
    
    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class BDS500DataModule(pl.LightningDataModule):
    
    def __init__(self,image_dir,
                 upscale_factor=2,
                 batch_size=64,
                 num_workers=4):
        
        super().__init__()
        self.image_dir = image_dir
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self,stage=None):
        self.train_ds = BDS500DataModule.get_training_set(upscale_factor=self.upscale_factor,
                                                              root_dir=self.image_dir)
        self.val_ds = BDS500DataModule.get_test_set(upscale_factor=self.upscale_factor,
                                                        root_dir=self.image_dir)
    
    def train_dataloader(self):
        self.training_data_loader = DataLoader(dataset=self.train_ds, 
                                               num_workers=self.num_workers, 
                                               batch_size=self.batch_size, 
                                               shuffle=True)
        return self.training_data_loader
    
    def val_dataloader(self):
        self.testing_data_loader = DataLoader(dataset=self.val_ds, 
                                              num_workers=self.num_workers,
                                              batch_size=self.batch_size, 
                                              shuffle=False)
        return self.testing_data_loader
    
    @staticmethod
    def calculate_valid_crop_size(crop_size, upscale_factor):
        return crop_size - (crop_size % upscale_factor)

    @staticmethod
    def input_transform(crop_size, upscale_factor):
        return Compose([
            CenterCrop(crop_size),
            Resize(crop_size // upscale_factor),
            ToTensor(),
        ])

    @staticmethod
    def target_transform(crop_size):
        return Compose([
            CenterCrop(crop_size),
            ToTensor(),
        ])

    @staticmethod
    def get_training_set(upscale_factor,root_dir):
        train_dir = os.path.join(root_dir, "train")
        crop_size = BDS500DataModule.calculate_valid_crop_size(256, upscale_factor)
        return DatasetFromFolder(train_dir,
                                 input_transform=BDS500DataModule.input_transform(crop_size, upscale_factor),
                                 target_transform=BDS500DataModule.target_transform(crop_size))

    @staticmethod
    def get_test_set(upscale_factor,root_dir):
        test_dir = os.path.join(root_dir, "test")
        crop_size = BDS500DataModule.calculate_valid_crop_size(256, upscale_factor)
        return DatasetFromFolder(test_dir,
                                 input_transform=BDS500DataModule.input_transform(crop_size, upscale_factor),
                                 target_transform=BDS500DataModule.target_transform(crop_size))