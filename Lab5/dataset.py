from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch

class clevr_dataset(Dataset):
    def __init__(self, img_path, json_path):
        """
        :param img_path: file of training images
        :param json_path: train.json
        """
        self.img_path = img_path
        with open(os.path.join('dataset', 'objects.json'), 'r') as file:
            self.classes = json.load(file)
        self.num_classes = len(self.classes)
        self.img_names = []
        self.img_conditions = []
        with open(json_path, 'r') as file:
            dict = json.load(file)
            for img_name, img_condition in dict.items():
                self.img_names.append(img_name)
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
        self.transformations = transforms.Compose([transforms.Resize((64,64)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        img = self.transformations(img)
        condition = self.int2onehot(self.img_conditions[index])
        return img, condition

    def int2onehot(self, int_list):
        onehot = torch.zeros(self.num_classes)
        for i in int_list:
            onehot[i] = 1.
        return onehot