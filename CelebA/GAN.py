import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Convolutional Generator for CelebA
    """
    def __init__(self, input_size=128, code_size=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size+code_size, 2*2*448)
        self.bn1 = nn.BatchNorm1d(2*2*448)
        self.conv1 = nn.ConvTranspose2d(448, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(3)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        
    def forward(self, z, c):
        z = z.view(z.size(0), -1)
        c = c.view(c.size(0), -1)
        noise = torch.cat((z, c), 1)
        x = self.fc1(noise)
        x = self.bn1(x)
        x = self.activation1(x)
        x = x.view(x.size(0), 448, 2, 2)
        x = self.conv1(x) # 2x2x448 -> 4x4x256
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.conv2(x) # 4x4x256 -> 8x8x128
        x = self.activation1(x)
        x = self.conv3(x) # 8x8x128 -> 16x16x64
        x = self.activation1(x)
        x = self.conv4(x) # 16x16x64 -> 32x32x3
        x = self.activation2(x)

        return x


class Discriminator(nn.Module):
    """
    Convolutional Discriminator for CelebA
    """
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*4*4, 1)
        self.activation2 = nn.Sigmoid()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y) # This is not done in the paper.
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.activation(y)

        d = self.fc1(y)
        d = self.activation2(d)  # Real / Fake

        return d, y # return with top layer features for Q


class Qrator(nn.Module):
    """
    Regularization Network for increasing Mutual Information in InfoGAN.

    This network takes as input the features from the discriminator's last hidden layer and outputs 
    three sets of codes:
    - c_discrete: a categorical code that represents the digit label (0-9) of the generated image.
    - c_mu: a continuous code that represents the mean values of the rotation and thickness of the generated image.
    - c_var: a continuous code that represents the variances of the rotation and thickness of the generated image.

    By maximizing the mutual information between the generated images and these codes, the generator can 
    learn to control these aspects of the generated images and produce more diverse and meaningful samples. 
    The Qrator network helps to estimate these codes from the discriminator's features, and the generator 
    is trained to maximize the mutual information between these codes and the generated images.
    """
    def __init__(self):
        super(Qrator, self).__init__()
        self.fc1 = nn.Linear(256*4*4, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.activation1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, 100)
        
    def forward(self, x):
        # Seperate code
        c = x.view(x.size(0), -1)
        c = self.fc1(x)
        c = self.bn1(c)
        c = self.activation1(c)
        c = self.fc2(c)

        c_discrete_list = []

        for i in range(0, 100, 10):
            c_slice = c[i:i+10]  # Slice the tensor to get 10 consecutive values
            c_discrete = torch.softmax(c_slice, dim=-1)  # Convert to discrete probability distribution
            c_discrete_list.append(c_discrete)

        # c_discrete = torch.softmax(c[:, :10], dim=-1) # Digit Label {0~9}
        # c_mu = c[:, 10:12] # mu & var of Rotation & Thickness
        # c_var = c[:, 12:14].exp() # mu & var of Rotation & Thickness
        return c_discrete_list
