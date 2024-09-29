import torch
import yaml
import argparse
import os
import numpy as np
from torch.optim import Adam
from tqdm import tqdm
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print("<--------------------Running diffusion model with following configuration--------------------->")
    print(config)
    print("<--------------------------------------------------------------------------------------------->")
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],beta_start=diffusion_config['beta_start'],beta_end=diffusion_config['beta_end'])
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    model = Unet(model_config).to(device)
    model.train()
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
        print('Loading checkpoint as found one')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],train_config['ckpt_name']), map_location=device))
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    for epoch_idx in range(num_epochs):
        losses = []
        batch_num = 0
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)
            noise = torch.randn_like(im).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_num += 1
            if batch_num % 100 == 0:
                print(f"Finished training on batch num {batch_num} in epoch {epoch_idx+1}")
        print('Finished epoch:{} | Loss : {:.4f}'.format(epoch_idx + 1,np.mean(losses),))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],train_config['ckpt_name']))
    print('... Done Training ...')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)