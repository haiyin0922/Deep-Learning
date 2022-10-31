import torch
import os
import json
import numpy as np
import random
from torchvision.utils import save_image
from sagan import Generator
from evaluator import evaluation_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def random_z(batch_size, z_dim):
    return torch.randn(batch_size, z_dim)

def get_test_conditions(path):
    """
    :return: (#test conditions,#classes) tensors
    """
    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        classes = json.load(file)
    with open(path, 'r') as file:
        test_conditions_list = json.load(file)

    labels = torch.zeros(len(test_conditions_list), len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i, int(classes[condition])] = 1.

    return labels

def main():
    set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Generator(100, 200)
    model.load_state_dict(torch.load('best models/test.pt'))
    model.to(device)

    evaluation = evaluation_model()
    test_conditions = get_test_conditions(os.path.join('dataset','test.json')).to(device)
    fixed_z = random_z(len(test_conditions), 100).to(device)

    with torch.no_grad():
        gen_imgs = model(fixed_z, test_conditions)

    score = evaluation.eval(gen_imgs, test_conditions)
    print(f'test score: {score:.2f}')
    save_image(gen_imgs, os.path.join('best models', f'test.png'), nrow=8, normalize=True)

    model.load_state_dict(torch.load('best models/new_test.pt'))
    model.to(device)

    test_conditions = get_test_conditions(os.path.join('dataset','new_test.json')).to(device)
    fixed_z = random_z(len(test_conditions), 100).to(device)

    with torch.no_grad():
        gen_imgs = model(fixed_z, test_conditions)

    score = evaluation.eval(gen_imgs, test_conditions)
    print(f'new_test score: {score:.2f}')
    save_image(gen_imgs, os.path.join('best models', f'new_test.png'), nrow=8, normalize=True)


if __name__ == '__main__':
    main()
