import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from matplotlib.pyplot import imsave

from WGAN import W_Discriminator, W_Generator
from utils import to_onehot, wget_sample_image


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

# Optimisers
D_opt = torch.optim.RMSprop(D.parameters(), lr=0.0005)
G_opt = torch.optim.RMSprop(G.parameters(), lr=0.0005)

# Training parameters
max_epoch = 100 # need more than 100 epochs for training generator
step = 0
g_step = 0
n_noise = 100

# Initialise the labels
D_labels = torch.ones([batch_size, 1]).to(DEVICE) # Discriminator Label to real
D_fakes = torch.zeros([batch_size, 1]).to(DEVICE) # Discriminator Label to fake

# Train WGAN
for epoch in range(max_epoch+1):
    for idx, (images, labels) in enumerate(dataloader):
        step += 1

        # Training Discriminator
        x = images.to(DEVICE)
        y = labels.view(batch_size, 1)
        y = to_onehot(y).to(DEVICE)
        x_outputs = D(x, y)

        z = torch.randn(batch_size, n_noise).to(DEVICE)
        z_outputs = D(G(z, y), y)
        
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
            z = torch.randn(batch_size, n_noise).to(DEVICE)
            z_outputs = D(G(z, y), y)
            
            # Wasserstein distance for the generator
            G_loss = -torch.mean(z_outputs)

            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_opt.step()
            
        # Log losses and histograms for tensorboard
        if step > 500 and step % LOG_LOSS_FREQ == 0:
            writer.add_scalar('loss/generator', G_loss, step)
            writer.add_scalar('loss/discriminator', D_loss, step)
        
        # Print losses every PRINT_LOSS_FREQ steps
        if step % PRINT_LOSS_FREQ == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item()))
        
        if step % SAVE_IMAGES_FREQ == 0:
            G.eval()
            img = wget_sample_image(G, DEVICE=DEVICE, n_noise=n_noise)
            imsave('samples/{}_step{}.jpg'.format(MODEL_NAME, str(step).zfill(3)), img, cmap='gray')
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
