import os
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from GAN import Discriminator, Generator, Mine
from utils import sample_noise, get_sample_image, fro_norm, adaptive_clip_grad


CHECKPOINT_DIR = './checkpoints'
MODEL_NAME = 'infoGAN'
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
M = Mine().to(DEVICE)

if not os.path.exists("data/MNIST/"):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                    std=[0.5])]
    )
    mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)

if not os.path.exists("samples/"):
    os.mkdir("samples/")


batch_size = 64
data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

D_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.99))
G_opt = torch.optim.Adam([{'params':G.parameters()}, {'params':M.parameters()}], lr=1e-3, betas=(0.5, 0.99))
#G_opt = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.99))

M_opt = torch.optim.Adam(M.parameters(), lr=1e-4)
max_epoch = 50 # need more than 200 epochs for training generator
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_noise = 62
n_c_discrete, n_c_continuous = 10, 2

D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake


#training MineGAN
for epoch in range(max_epoch+1):
    for idx, (images, labels) in enumerate(data_loader):
        step += 1
        labels = labels.view(batch_size, 1)
        # Training Discriminator
        x = images.to(DEVICE)
        x_outputs, _, = D(x)
        D_x_loss = bce_loss(x_outputs, D_labels)

        z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)
        z_outputs, _, = D(G(z, c))
        D_z_loss = bce_loss(z_outputs, D_fakes)
        D_loss = D_x_loss + D_z_loss
        
        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()

        # Training Generator
        z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)


         # (B,1), (B,10), (B,4)
        gen_output = G(z, c)
        z_outputs, features = D(gen_output) 

        G_loss = bce_loss(z_outputs, D_labels)  #D_labels

        _, c_bar = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)

        # Sampling from joint
        M_c_disc, M_c_cont = M(gen_output, c)
        # Sampling from marginals
        M_c_bar_disc, M_c_bar_cont = M(gen_output, c_bar)
        # Calculating MINE
        mi_discrete = torch.mean(M_c_disc) - torch.log(torch.mean(torch.exp(M_c_bar_disc)+1e-8))
        mi_cont1 = torch.mean(M_c_cont[0]) - torch.log(torch.mean(torch.exp(M_c_bar_cont[0])+1e-8))
        mi_cont2 = torch.mean(M_c_cont[1]) - torch.log(torch.mean(torch.exp(M_c_bar_cont[1])+1e-8))
        mutual_info_loss = -1*(mi_discrete + mi_cont1 + mi_cont2)
        
       
        if -1*mutual_info_loss.item() > 20:
          GnQ_loss = G_loss
        else:
          GnQ_loss = G_loss + mutual_info_loss

        
        # Gradient clipping needed to prevent MINE blowup
        G_opt.zero_grad()
        GnQ_loss.backward()
        G_grad_norm = fro_norm(G)
        MI_grad_norm = fro_norm(M)
        adaptive_clip_grad(M, G_grad_norm, MI_grad_norm)
        # Optional grad clip
        #clip_grad(G)
        G_opt.step()
        
        if step % 500 == 0:
            print(MI_grad_norm.item())
            print(f"discrete: {mi_discrete}")
            print(f"cont 1: {mi_cont1}, cont 2: {mi_cont2}")
            print('Epoch: {}/{}, Step: {}, Mut info: {}, D Loss: {}, G Loss: {}, GnQ Loss: {}, Time: {}'\
                  .format(epoch, max_epoch, step, -1*mutual_info_loss.item(), D_loss.item(), G_loss.item(), GnQ_loss.item(), str(datetime.datetime.today())[:-7]))
            
        if step % 1000 == 0:
            G.eval()
            img1, img2, img3 = get_sample_image()
            imsave('samples/{}_step{}_type1.jpg'.format(MODEL_NAME, str(step).zfill(3)), img1, cmap='gray')
            imsave('samples/{}_step{}_type2.jpg'.format(MODEL_NAME, str(step).zfill(3)), img2, cmap='gray')
            imsave('samples/{}_step{}_type3.jpg'.format(MODEL_NAME, str(step).zfill(3)), img3, cmap='gray')
            G.train()
