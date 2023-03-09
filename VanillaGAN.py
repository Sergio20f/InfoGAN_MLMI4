import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Convolutional Generator for MNIST
    """
    def __init__(self, input_size=62,  num_classes=784):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 7*7*128)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        
    def forward(self, z):
        noise = z.view(z.size(0), -1)

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


class Discriminator(nn.Module):
    """
    Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.activation2 = nn.Sigmoid()
    
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
        y = self.activation(y)
        d = self.fc2(y)
        d = self.activation2(d)  # Real / Fake 

        return d # return with top layer features for Q

