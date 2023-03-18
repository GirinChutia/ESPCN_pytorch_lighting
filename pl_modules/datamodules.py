import pytorch_lightning as pl
import torch.utils.data as data
import os
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, InterpolationMode
from torchvision.transforms import Normalize, GaussianBlur
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
import numpy as np

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, 
                 image_dir,
                 input_transform=None, 
                 target_transform=None,
                 upscale_factor=3,
                 crop_size=256,
                 stage=None):
        
        super(DatasetFromFolder, self).__init__()
        input_crop_size = crop_size
        crop_size = calculate_valid_crop_size(crop_size,upscale_factor)
        
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) \
            if DatasetFromFolder.is_image_file(x)]
        
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        # if stage == 'train':
        #     self.crop_transform = A.Compose([A.RandomCrop(width=crop_size, height=crop_size),
        #                                     A.HorizontalFlip(p=0.25)])
        # else:
        #     self.crop_transform = A.Compose([A.RandomCrop(width=crop_size, height=crop_size)])
        
        if stage == 'train':
            self.crop_transform = A.Compose([A.CenterCrop(width=crop_size, height=crop_size,always_apply=True),
                                            A.HorizontalFlip(p=0.25)])
        else:
            self.crop_transform = A.Compose([A.CenterCrop(width=crop_size, height=crop_size,always_apply=True)])
            
        print(f'Input Crop size : {input_crop_size}')
        print(f'Valid Crop size : {crop_size}, Upscale factor : {upscale_factor}')

    def __getitem__(self, index):
        pilimg = Image.open(self.image_filenames[index]) #RGB
        npimg = np.array(pilimg)
        trans = self.crop_transform(image=npimg)
        trans_npimg = trans["image"] # RGB
        _input = Image.fromarray(trans_npimg,mode='RGB') # Input
        _inputpilimgYCBCR = _input.convert('YCbCr')
        input,_,_ = _inputpilimgYCBCR.split() # Convert RGB to YCbCr --> Paper uses only Y Channel
        target = input.copy() # Target
        
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        return input, target

    def __len__(self):
        return len(self.image_filenames)
    
    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class BDSDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 image_dir,
                 upscale_factor=3,
                 batch_size=64,
                 num_workers=4,
                 crop_size=256,
                 normalize=False,
                 means=[],
                 stds=[]):
        
        super().__init__()
        self.image_dir = image_dir
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.normalize = normalize
        self.means = means
        self.stds = stds
    
    def prepare_data(self) -> None:
        pass
    
    @staticmethod
    def target_transform(normalize=False,means=[],stds=[]):
        if normalize:
            return Compose([ToTensor(),Normalize(mean=means,std=stds)])
        else:
            return Compose([ToTensor()])
    
    @staticmethod
    def input_transform(crop_size, upscale_factor,normalize=False,means=[],stds=[]):
        if normalize:
            return Compose([Resize(crop_size // upscale_factor,
                                   interpolation=InterpolationMode.BICUBIC),
                            GaussianBlur(kernel_size=3,sigma=0.5),
                            ToTensor(),
                            Normalize(mean=means,std=stds)])
        else:
            return Compose([Resize(crop_size // upscale_factor,
                                interpolation=InterpolationMode.BICUBIC),
                            ToTensor()])
        
    @staticmethod
    def get_training_set(upscale_factor,root_dir,crop_size,
                         normalize=False,means=[],stds=[]):
        train_dir = os.path.join(root_dir, "train")
        return DatasetFromFolder(train_dir,
                                 input_transform=BDSDataModule.input_transform(crop_size, upscale_factor,
                                                                               normalize=normalize,means=means,stds=stds),
                                 target_transform=BDSDataModule.target_transform(normalize=normalize,means=means,stds=stds),
                                 stage='train')
        
    @staticmethod
    def get_val_set(upscale_factor,root_dir,crop_size,
                    normalize=False,means=[],stds=[]):
        val_dir = os.path.join(root_dir, "val")
        return DatasetFromFolder(val_dir,
                                 input_transform=BDSDataModule.input_transform(crop_size, upscale_factor,
                                                                               normalize=normalize,means=means,stds=stds),
                                 target_transform=BDSDataModule.target_transform(normalize=normalize,means=means,stds=stds),
                                 stage='val')
        
    @staticmethod
    def get_test_set(upscale_factor,root_dir,crop_size,
                     normalize=False,means=[],stds=[]):
        test_dir = os.path.join(root_dir, "test")
        return DatasetFromFolder(test_dir,
                                 input_transform=BDSDataModule.input_transform(crop_size, upscale_factor,
                                                                               normalize=normalize,means=means,stds=stds),
                                 target_transform=BDSDataModule.target_transform(normalize=normalize,means=means,stds=stds),
                                 stage='test')
    
    def setup(self,stage=None):
        print(f'Use Normalization : {self.normalize}')
        self.train_ds = BDSDataModule.get_training_set(upscale_factor=self.upscale_factor,
                                                       root_dir=self.image_dir,
                                                       crop_size=self.crop_size,
                                                       normalize=self.normalize,
                                                       means=self.means,
                                                       stds=self.stds)
        
        self.val_ds = BDSDataModule.get_val_set(upscale_factor=self.upscale_factor,
                                                root_dir=self.image_dir,
                                                crop_size=self.crop_size,
                                                normalize=self.normalize,
                                                means=self.means,
                                                stds=self.stds)
        
        self.test_ds = BDSDataModule.get_test_set(upscale_factor=self.upscale_factor,
                                                 root_dir=self.image_dir,
                                                 crop_size=self.crop_size,
                                                 normalize=self.normalize,
                                                 means=self.means,
                                                 stds=self.stds)
        
    def train_dataloader(self):
        self.data_loader = DataLoader(dataset=self.train_ds, 
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size, 
                                      shuffle=True)
        return self.data_loader
    
    def val_dataloader(self):
        self.data_loader = DataLoader(dataset=self.val_ds, 
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size, 
                                      shuffle=False)
        return self.data_loader
    
    def test_dataloader(self):
        self.data_loader = DataLoader(dataset=self.test_ds, 
                                      num_workers=self.num_workers,
                                      batch_size=self.batch_size, 
                                      shuffle=False)
        return self.data_loader
