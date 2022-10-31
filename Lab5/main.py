import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from dataset import clevr_dataset
from sagan import Generator, Discriminator
from train import train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    c_dim = 200
    image_shape = (64, 64, 3)
    epochs = 300
    lr = 1e-4
    batch_size = 64
    set_seed(1)

    # load training data
    dataset_train = clevr_dataset(img_path='iclevr', json_path=os.path.join('dataset','train.json'))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    # create generate & discriminator
    generator = Generator(z_dim, c_dim).to(device)
    discrimiator = Discriminator(image_shape).to(device)
    #generator.weight_init(mean=0, std=0.02) # dcgan
    #discrimiator.weight_init(mean=0, std=0.02) # dcgan

    # train
    train(loader_train, generator, discrimiator, z_dim, epochs, lr)

if __name__ == '__main__':
    main()