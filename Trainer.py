import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader import Dataset_Game
from torchvision.utils import save_image
from torchvision.models import resnet50, ResNet50_Weights
import random
import torch.optim as optim
from torch import stack
import json
from tqdm import tqdm

import matplotlib.pyplot as plt


class Reg_model(nn.Module):
    def __init__(self, args):
        super(Reg_model, self).__init__()
        self.args = args
        
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_ftr = self.resnet.fc.in_features
        out_ftr = 1
        self.resnet.fc = nn.Linear(in_ftr, out_ftr, bias=True)
        self.optim      = optim.Adam(self.resnet.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4, 8, 15], gamma=0.1)
        self.ce_criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.writer = SummaryWriter(args.save_root)
        self.batch_size = args.batch_size
        self.device = args.device
        
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            # print(len(train_loader))
            total_loss = 0
            pbar = tqdm(train_loader, ncols=120)
            for img, label in pbar:
                img = img.to(self.device)
                label = label.to(self.device)
                # training one step
                self.optim.zero_grad()
                result = self.resnet(img)
                #result = torch.clamp(result, -1, 600000)
                loss = self.mae_loss(result, label.float())
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
                
                self.tqdm_bar('train ', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            print('training loss:', total_loss/len(train_loader))
            self.writer.add_scalar('train/loss', total_loss/len(train_loader), self.current_epoch)
            if (self.current_epoch+1) % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch_{self.current_epoch}.ckpt"))
                
            eval_acc = self.eval()
            self.writer.add_scalar('val/loss', eval_acc, self.current_epoch)
            self.current_epoch += 1
            self.scheduler.step()
    
    def store_parameters(self):
        # save args
        with open(os.path.join(self.args.save_root, 'args.yaml'), 'w') as f:
            for k, v in vars(self.args).items():
                f.write(f"{k}: {v}\n")

            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_loss = 0
        pbar = tqdm(val_loader, ncols=120)
        with torch.no_grad():
            for (img, label) in pbar:
                img = img.to(self.device)
                label = label.to(self.device)
                result = self.resnet(img)
                loss = self.mae_loss(result, label.float())
                total_loss += loss.item()
                self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        print(f"Epoch {self.current_epoch}, val_loss: {total_loss/len(val_loader)}")
        return total_loss/len(val_loader)
    
    def mae_loss(self, outputs, labels):
        criterion = nn.MSELoss()
        # print(outputs)
        return criterion(outputs, labels.unsqueeze(-1))  
    
    def train_dataloader(self):
        dataset = Dataset_Game(root=self.args.DR, mode='train')
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)  
        return train_loader
    
    def val_dataloader(self):
        dataset = Dataset_Game(root=self.args.DR, mode='val')  
        val_loader = DataLoader(dataset,
                                  batch_size=16,
                                  num_workers=self.args.num_workers,
                                  shuffle=False)  
        return val_loader
    
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.current_epoch = checkpoint['last_epoch']




def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Reg_model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()
    
    test_dataset = Dataset_Game(root=args.DR, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    mses = 0
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, ncols=120):
            scores = []
            for i in range(imgs.shape[0]):
                img = imgs[i].unsqueeze(0).to(args.device)
                output = model.resnet(img)
                scores.append(output.cpu().detach().numpy().item())
            average = np.mean(scores)
            mses += (average - labels[0].item()) ** 2
    mses /= len(test_loader)



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(add_help=True)
#     parser.add_argument('--batch_size',    type=int,    default=2)
#     parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
#     parser.add_argument('--test',          action='store_true')
#     parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
#     parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
#     parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
#     parser.add_argument('--num_workers',   type=int, default=4)
#     parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
#     parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
#     parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
#     parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
#     parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    

    

#     args = parser.parse_args()
    
#     main(args)
