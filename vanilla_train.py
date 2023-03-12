import os
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from VanillaGAN import Discriminator, Generator
from utils import vanilla_gan_get_sample_image


CHECKPOINT_DIR = './vanilla_gan_checkpoints'
MODEL_NAME = 'VanillaGAN'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_IMAGES_FREQ = 1000 # 1000 originaly
PRINT_LOSS_FREQ = 500 # 500 originaly
LOG_LOSS_FREQ = 100 # 100 originaly
SAVE_CHECKPOINT_FREQ = 1000

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    
writer = SummaryWriter()

D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)

if not os.path.exists("data/MNIST/"):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                    std=[0.5])]
    )
    mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)

if not os.path.exists("vanilla_gan_samples/"):
    os.mkdir("vanilla_gan_samples/")

batch_size = 128
dataloader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# Losses
criterion = nn.BCELoss()

# Optimisers
D_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.99))
G_opt = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.99))

# Training parameters
max_epoch = 50
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_noise = 100


# Initialise the labels
D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

# Train the GAN model
for epoch in range(max_epoch):
    for idx, (images, _) in enumerate(dataloader):
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels)

        z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        if step % n_critic == 0:
            # Training Generator
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            G.zero_grad()
            G_loss.backward()
            G_opt.step()
        
        if step % 500 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))
        
        if step % 1000 == 0:
            G.eval()
            img = vanilla_gan_get_sample_image(G, n_noise)
            plt.imsave('samples/{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')
            G.train()
        step += 1


writer.export_scalars_to_json("./all_summary.json")
writer.close()
