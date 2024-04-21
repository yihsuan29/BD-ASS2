
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
import json

from torchvision import transforms

class Dataset_Game(torchData):
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
        
        img_h = ? # resize 256
        img_w = ?
        game_number = ? # number of games in json file
        
        self.all_images = []
        self.all_labels = []
        
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.RandomHorizontalFlip(),      # data augmentation
                transforms.RandomRotation(-10, 10),     # data augmentation
                transforms.ToTensor()
            ])
            games_json = games_json[:int(game_number*0.6)]
        elif mode == 'val':
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
            games_json = games_json[int(game_number*0.6):int(game_number*0.8)]
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
            games_json = games_json[int(game_number*0.8):]
        
        self.get_img_paths(games_json)
        
    def get_img_paths(self, json_file):
        # iterate through json file, read the paths and the images, return them
        for game in json_file:
            for frame in game['screeshots']:
                # read image from the path directly
                frame = frame[0:-3] + 'webp'
                self.all_images.append(...) # read jpg image
                
                # TODO
                self.all_labels.append(int(game['price']))
        

    def __len__(self):
        return len(self.img_folder)

    def __getitem__(self, index):
        return self.all_images[index], self.all_labels[index]
    
    

