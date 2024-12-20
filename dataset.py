import os
import torch
import torchvision
from torch.utils.data import Dataset
import json

class CombinedImgDataset(Dataset):
    def __init__(self, dataset_path):
        # folder should contain dataset.json file with dataset description
        self.path = dataset_path
        
        self.img_paths = []
        self.labels = []
        with open(self.path + '/dataset.json', "r") as file:
            f = json.load(file)
            for obj in f['labels']:
                self.img_paths.append(obj[0])
                self.labels.append(obj[1])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, item):
        img_name = self.path + '/' + self.img_paths[item]
        img = torchvision.io.read_image(img_name)
        img = (img - 127.5) / 127.5
        return img, self.labels[item]

    
class SourceImgDataset(Dataset):
    def __init__(self, dataset_path, lbl_val=1):
        # folder should contain dataset.json file with dataset description
        self.path = dataset_path
        
        self.img_paths = []
        with open(self.path + '/dataset.json', "r") as file:
            f = json.load(file)
            for obj in f['labels']:
                if obj[1] == lbl_val:
                    self.img_paths.append(obj[0])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, item):
        img_name = self.path + '/' + self.img_paths[item]
        img = torchvision.io.read_image(img_name)
        img = (img - 127.5) / 127.5
        return img