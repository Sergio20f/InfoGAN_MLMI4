import os
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from GAN import Discriminator, Generator, Qrator
from utils import sample_noise, log_gaussian, get_sample_image

CHECKPOINT_DIR = './checkpoints'
MODEL_NAME = "infoWGAN"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_IMAGES_FREQ = 1000 # 1000 originaly
PRINT_LOSS_FREQ = 500 # 500 originaly
LOG_LOSS_FREQ = 100 # 100 originaly
SAVE_CHECKPOINT_FREQ = 1000

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    
writer = SummaryWriter()

### CHANGE WITH WGAN NETWORKS
D = Discriminator().to(DEVICE)
G = Generator().to(DEVICE)
Q = Qrator().to(DEVICE)

if not os.path.exists("data/MNIST/"):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5],
                                    std=[0.5])]
    )
    mnist = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)

if not os.path.exists("samples/"):
    os.mkdir("samples/")

batch_size = 128
dataloader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

# Loss for Q
ce_loss = nn.CrossEntropyLoss()

# Optimisers
D_opt = torch.optim.RMSprop(D.parameters(), lr=5e-5)
G_opt = torch.optim.RMSprop(G.parameters(), lr=5e-5)
Q_opt = torch.optim.Adam(Q.parameters(), lr=1e-3, betas=(0.5, 0.99))
# Probably will have to change due to the fact that G uses W_d and Q BCE

# Training parameters
max_epoch = 50
step = 0
n_critic = 1 # for training more k steps about Discriminator
n_noise = 62
n_c_discrete, n_c_continuous = 10, 2
lambda_gp = 10

# Initialise the labels
D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

# Train InfoWGAN model
for epoch in range(max_epoch+1):
    for idx, (images, labels) in enumerate(dataloader):
        step += 1

        # Reshape labels to match batch size
        labels = labels.view(batch_size, 1)

        # Training critic
        for k in range(n_critic):
            x = images.to(DEVICE)
            x_outputs, _ = D(x)
            D_x_loss = -torch.mean(x_outputs) # New loss - it doesn't account for labels

            z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)
            z_outputs, _ = D(G(z, c)) # detach the G?
            D_z_loss = torch.mean(z_outputs) # New loss - it doesn't account for fake labels
            # Compute the Wasserstein distance
            D_loss = D_x_loss + D_z_loss

            # Gradient penalty - ensures Lipschitz constraint on the discriminator
            alpha = torch.rand(batch_size, 1, 1, 1).to(DEVICE)
            interpolates = (alpha * x + (1 - alpha) * G(z, c)).requires_grad_(True)
            d_interpolates = D(interpolates)
            gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones_like(d_interpolates),
                                            create_graph=True, retain_graph=True)[0]
            gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1)**2)
            # Update loss
            D_loss += lambda_gp * gradient_penalty

            # Backprop
            D_opt.zero_grad()
            D_loss.backward()
            D_opt.step()

            # Clip discriminator weights
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        # Training the Generator
        z, c = sample_noise(batch_size, n_noise, n_c_discrete, n_c_continuous, label=labels, supervised=True)
        z_outputs, features = D(G(z, c))
        
        # Predict the latent codes with the Qrator using discriminator features
        c_discrete_out, cc_mu, cc_var = Q(features)

        # Get discrete label from continuous vector using argmax
        c_discrete_label = torch.max(c[:, :-2], 1)[1].view(-1, 1)

        # Calculate Wasserstein distance for the Generator network
        G_loss = -torch.mean(z_outputs)

        # Calculate cross entropy loss for discrete code
        Q_loss_discrete = ce_loss(c_discrete_out, c_discrete_label.view(-1))

        # Calculate log-likelihood of continuous code assuming Gaussian distribution
        Q_loss_continuous = -torch.mean(torch.sum(log_gaussian(c[:, -2:], cc_mu, cc_var), 1))

        # Calculate total mutual information loss
        mutual_info_loss = Q_loss_discrete + Q_loss_continuous*0.1

        # Calculate total loss for Generator and Qrator
        GnQ_loss = G_loss + mutual_info_loss

        G_opt.zero_grad()
        GnQ_loss.backward() # This or G_loss
        G_opt.step()

        Q_opt.zero_grad()
        mutual_info_loss.backward() # This or GnQ loss?
        Q_opt.step()


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