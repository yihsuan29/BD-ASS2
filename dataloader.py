
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
import json

from torchvision import transforms

def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')
    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, mode='train'):
        super().__init__()
        assert mode in ['train', 'val', 'test'], "There is no such mode !!!"
        # root is the path of JSON file
        with open(root, 'r') as file:
            games_json = json.load(file)
        
        img_h = ?
        img_w = ?
        game_number = ? # number of games in json file
        
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.RandomHorizontalFlip(),      # data augmentation
                transforms.RandomRotation(-10, 10),     # data augmentation
                transforms.ToTensor()
            ])
            
        elif mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
        
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        
        # get a random seed
        new_seed = torch.randint(0, 100000, (1,)).item()
        
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('/')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[(index*self.video_len)+i]
            label_name = '/'.join(label_list)
            torch.manual_seed(new_seed)
            imgs.append(self.transform(imgloader(img_name)))
            torch.manual_seed(new_seed)
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)
    
    

