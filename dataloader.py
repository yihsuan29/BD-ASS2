
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
import json
import numpy as np
from torchvision import transforms

class Dataset_Game(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, mode='train',task='Cla'):
        super().__init__()
        assert mode in ['train', 'val', 'test'], "There is no such mode !!!"
        self.root = root
        self.mode = mode
        self.task = task
        root = os.path.join(root, 'dataset.json')
        # root is the path of JSON file
        with open(root, 'r') as file:
            games_json = json.load(file)
        
        img_h = 256
        img_w = 512
        
        
        self.all_images = []
        self.all_labels = []
        
        
        self.game_number = 0
        self.get_img_paths(games_json,)
        
        if mode == 'train':
            self.transform = transforms.Compose([
                #transforms.Resize((img_h, img_w)),
                #transforms.RandomHorizontalFlip(),      # data augmentation
                #transforms.RandomRotation(-10, 10),     # data augmentation
                transforms.ToTensor()
            ])
            games_json = games_json[:int(self.game_number*0.6)]
        elif mode == 'val':
            self.transform = transforms.Compose([
                #transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
            games_json = games_json[int(self.game_number*0.6):int(self.game_number*0.8)]
        else:
            self.transform = transforms.Compose([
                #transforms.Resize((img_h, img_w)),
                transforms.ToTensor()
            ])
            games_json = games_json[int(self.game_number*0.8):]
        
    def get_img_paths(self, json_file):
        # iterate through json file, read the paths and the images, return them
        for game in json_file:
            if self.task == "Cla" and game.get('sentiment') is None:
                continue
            self.game_number += 1
            if self.mode != 'test':
                for frame in game['screenshots']:
                    # read image from the path directly
                    frame = frame[0:-3] + 'webp'
                    frame = os.path.join(self.root, 'imgori', frame)
                    # image = imgloader(frame)
                    # image = self.transform(image)
                    self.all_images.append(frame) # read jpg image
                if self.task == "Cla":
                    sentiment_label = self.map_sentiment(game['sentiment'])
                    self.all_labels.append(sentiment_label)
                else:
                    self.all_labels.append(int(game['price']))
            else:
                temp_img = []
                temp_label = []
                for frame in game['screenshots']:
                    frame = frame[0:-3] + 'webp'
                    frame = os.path.join(self.root, 'imgori', frame)
                    image = imgloader(frame)
                    image = self.transform(image)
                    temp_img.append(image)
                    if self.task == "Cla":
                        sentiment_label = self.map_sentiment(game['sentiment'])
                        temp_label.append(sentiment_label)
                    else:
                        temp_label.append(int(game['price']))
                self.all_images.append(temp_img)
                self.all_labels.append(temp_label)
        

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        img = imgloader(self.all_images[index])
        img = self.transform(img)
        img /= 255.0
        return img, self.all_labels[index]
    
    def map_sentiment(self, sentiment):
        SENTIMENT_MAP = {
            'Overwhelmingly Positive': 8,
            'Very Positive': 7,
            'Positive': 6,
            'Mostly Positive': 5,
            'Mixed': 4,
            'Mostly Negative': 3,
            'Negative': 2,
            'Very Negative': 1,
            'Overwhelmingly Negative': 0
        }
        return SENTIMENT_MAP.get(sentiment, 0)    
    
    

