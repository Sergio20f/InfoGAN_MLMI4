import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from WGAN import W_Discriminator, W_Generator, W_Qrator#, Q_test
from utils import to_onehot_2, sample_noise, log_gaussian, get_sample_image


CHECKPOINT_DIR = './checkpoints'
MODEL_NAME = 'W-GAN'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_IMAGES_FREQ = 1000 # 1000 originaly
PRINT_LOSS_FREQ = 500 # 500 originaly
LOG_LOSS_FREQ = 100 # 100 originaly
SAVE_CHECKPOINT_FREQ = 1000

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

writer = SummaryWriter()

D = W_Discriminator().to(DEVICE)
G = W_Generator().to(DEVICE)
Q = W_Qrator().to(DEVICE)
#Q_new = Q_test().to(DEVICE)

# Preprocessing images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                std=[0.5])])

if not os.path.exists("data/MNIST/"):
    mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
else:
    mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=False)

if not os.path.exists("samples/"):
    os.mkdir("samples/")

batch_size = 128
dataloader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# Losses
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

# Optimisers
D_opt = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0., 0.9))
G_opt = torch.optim.Adam([{'params':G.parameters()}, {'params':Q.parameters()}], lr=1e-4, betas=(0., 0.9))

# Training parameters
max_epoch = 100 # need more than 100 epochs for training generator
step = 0
g_step = 0
n_noise = 62
n_c_discrete, n_c_continuous = 10, 2
lambda_gp = 10

# Initialise the labels
D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

# Train WGAN
for epoch in range(max_epoch+1):
    for idx, (images, labels) in enumerate(dataloader):
        step += 1

        # Training Discriminator
        x = images.to(DEVICE)
        labels = labels.view(batch_size, 1)
        y = to_onehot_2(labels).to(DEVICE)
        x_outputs, _ = D(x, y)

        z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)
        #z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs, _ = D(G(z, c), y)
        
        # Wasserstein distance for the discriminator
        D_x_loss = torch.mean(x_outputs)
        D_z_loss = torch.mean(z_outputs)
        D_loss = D_z_loss - D_x_loss

        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        # Parameter (Weight) Clipping for K-Lipshitz constraint
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)
                    
        if step % 1 == 0:
            g_step += 1
            # Training Generator
            #z = torch.randn(batch_size, n_noise).to(DEVICE)
            z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)
            z_outputs, features = D(G(z, c), y)

            # Get discrete label from continuous vector using argmax
            c_discrete_label = torch.max(c[:, :-2], 1)[1].view(-1, 1)

            #print(features.shape)
            c_discrete_out, cc_mu, cc_var = Q(features)
            
            # Wasserstein distance for the generator
            G_loss = -torch.mean(z_outputs)

            # Calculate cross entropy loss for discrete code
            Q_loss_discrete = ce_loss(c_discrete_out, c_discrete_label.view(-1))

            # Calculate log-likelihood of continuous code assuming Gaussian distribution
            Q_loss_continuous = -torch.mean(torch.sum(log_gaussian(c[:, -2:], cc_mu, cc_var), 1))

            # Calculate total mutual information loss
            mutual_info_loss = Q_loss_discrete + Q_loss_continuous*0.1

            # Calculate total loss for Generator and Qrator
            GnQ_loss = G_loss + 1*mutual_info_loss # lambda

            D.zero_grad()
            G.zero_grad()
            GnQ_loss.backward()
            G_opt.step()
            
        # Log losses and histograms for tensorboard
        if step > 500 and step % LOG_LOSS_FREQ == 0:
            writer.add_scalar('loss/total', GnQ_loss, step)
            writer.add_scalar('loss/Q_discrete', Q_loss_discrete, step)
            writer.add_scalar('loss/Q_continuous', Q_loss_continuous, step)
            writer.add_scalar('loss/Q', mutual_info_loss, step)
            writer.add_histogram('output/mu', cc_mu)
            writer.add_histogram('output/var', cc_var)
        
        # Print losses every PRINT_LOSS_FREQ steps
        if step % PRINT_LOSS_FREQ == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}, GnQ Loss: {}, Time: {}'\
                  .format(epoch, max_epoch, step, D_loss.item(), G_loss.item(), GnQ_loss.item(), str(datetime.datetime.today())[:-7]))

        # Save generated images every SAVE_IMAGES_FREQ steps
        if step % SAVE_IMAGES_FREQ == 0:
            G.eval()
            img1, img2, img3 = get_sample_image(n_noise, n_c_continuous, G)
            plt.imsave('samples/{}_step{}_type1.jpg'.format(MODEL_NAME, str(step).zfill(3)), img1, cmap='gray')
            plt.imsave('samples/{}_step{}_type2.jpg'.format(MODEL_NAME, str(step).zfill(3)), img2, cmap='gray')
            plt.imsave('samples/{}_step{}_type3.jpg'.format(MODEL_NAME, str(step).zfill(3)), img3, cmap='gray')
            G.train()

        # Save model checkpoint every SAVE_CHECKPOINT_FREQ steps
        if step % SAVE_CHECKPOINT_FREQ == 0:
          checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_step{step}.pt')
          torch.save({
                'epoch': epoch,
                'step': step,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'G_opt_state_dict': G_opt.state_dict(),
                'D_opt_state_dict': D_opt.state_dict(),
            }, checkpoint_path)
          
writer.export_scalars_to_json("./all_summary.json")
writer.close()
