import os
import torch
import torch.nn as nn
import copy
import json
from torchvision.utils import save_image
from evaluator import evaluation_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, g_model, d_model, z_dim, epochs, lr):
    """
    :param z_dim: 100
    """
    Criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(g_model.parameters(), lr, betas=(0.5,0.99))
    optimizer_d = torch.optim.Adam(d_model.parameters(), lr, betas=(0.5,0.99))
    evaluation = evaluation_model()

    test_conditions = get_test_conditions(os.path.join('dataset','test.json')).to(device)
    fixed_z = random_z(len(test_conditions), z_dim).to(device)
    best_score = 0

    new_test_conditions = get_test_conditions(os.path.join('dataset','new_test.json')).to(device)
    new_fixed_z = random_z(len(new_test_conditions), z_dim).to(device)
    new_best_score = 0

    for epoch in range(1, 1+epochs):
        total_loss_g = 0
        total_loss_d = 0
        for (images, conditions) in dataloader:
            g_model.train()
            d_model.train()
            batch_size = len(images)
            images = images.to(device)
            conditions = conditions.to(device)

            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            """
            train discriminator
            """
            optimizer_d.zero_grad()

            # for real images
            predicts = d_model(images, conditions)
            #loss_real = Criterion(predicts, real) # dcgan
            loss_real = torch.nn.ReLU()(1.0 - predicts).mean() # sagan
            # for fake images
            z = random_z(batch_size, z_dim).to(device)
            gen_imgs = g_model(z, conditions)
            predicts = d_model(gen_imgs.detach(), conditions)
            #loss_fake = Criterion(predicts, fake) # dcgan
            loss_fake = torch.nn.ReLU()(1.0 + predicts).mean() # sagan
            # bp
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            """
            train generator
            """
            for _ in range(5):
                optimizer_g.zero_grad()

                z = random_z(batch_size, z_dim).to(device)
                gen_imgs = g_model(z, conditions)
                predicts = d_model(gen_imgs, conditions)
                #loss_g = Criterion(predicts, real) # dcgan
                loss_g = -predicts.mean() # sagan
                # bp
                loss_g.backward()
                optimizer_g.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

        # evaluate
        g_model.eval()
        d_model.eval()
        with torch.no_grad():
            gen_imgs = g_model(fixed_z, test_conditions)
            new_gen_imgs = g_model(new_fixed_z, new_test_conditions)
        score = evaluation.eval(gen_imgs, test_conditions)
        new_score = evaluation.eval(new_gen_imgs, new_test_conditions)
        if score > best_score:
            best_score = score
            best_model_wts = copy.deepcopy(g_model.state_dict())
            torch.save(best_model_wts, os.path.join('models', f'epoch{epoch}_{score:.2f}.pt'))
        if new_score > new_best_score:
            new_best_score = new_score
            best_model_wts = copy.deepcopy(g_model.state_dict())
            torch.save(best_model_wts, os.path.join('models', f'epoch{epoch}_{new_score:.2f}_new.pt'))

        print(f'[epoch{epoch}] loss_g: {total_loss_g/len(dataloader):.3f}  loss_d: {total_loss_d/len(dataloader):.3f}  testing score: {score:.2f}  new testing score: {new_score:.2f}')
        # savefig
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow=8, normalize=True)

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
