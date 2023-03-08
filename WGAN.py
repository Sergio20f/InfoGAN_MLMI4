import torch
import torch.nn as nn


class W_Generator(nn.Module):
    """
    Convolutional Generator for MNIST
    """
    def __init__(self, input_size=62, code_size=12, num_classes=784):
        super(W_Generator, self).__init__()
        self.fc1 = nn.Linear(input_size+code_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 7*7*128)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        
    def forward(self, z, c):
        z = z.view(z.size(0), -1)
        c = c.view(c.size(0), -1)
        noise = torch.cat((z, c), 1)
        x = self.fc1(noise)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv1(x)
        x = self.bn3(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)

        return x
    

class W_Discriminator(nn.Module):
    """
    Convolutional Discriminator for InfoWGAN
    """
    def __init__(self, in_channel=1):
        super(W_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.activation(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.bn3(y)
        y = self.activation(y) # y == FEATURES
        d = self.fc2(y)
        return d, y # return with top layer features for Q


class W_Qrator(nn.Module):
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
        super(W_Qrator, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.activation1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, 14)
        
    def forward(self, x):
        # Seperate code
        c = self.fc1(x)
        c = self.bn1(c)
        c = self.activation1(c)
        c = self.fc2(c.detach())
        c_discrete = torch.softmax(c[:, :10], dim=-1) # Digit Label {0~9}
        c_mu = c[:, 10:12] # mu & var of Rotation & Thickness
        c_var = c[:, 12:14].exp() # mu & var of Rotation & Thickness
        return c_discrete, c_mu, c_var