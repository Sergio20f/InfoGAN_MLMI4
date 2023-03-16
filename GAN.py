import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
        Convolutional Discriminator for MNIST
    """
    def __init__(self, in_channel=1):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(# 28 -> 14
            nn.Conv2d(in_channel, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.layer2 = nn.Sequential(# 14 -> 7
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        self.layer3 = nn.Sequential(#
            nn.Linear(128*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    
    def forward(self, x):
        y_ = self.layer1(x)
        y_ = self.layer2(y_)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.layer3(y_)
        d = self.fc(y_) # Real / Fake        
        return d, y_ # return with top layer features for M


class Generator(nn.Module):
    """
        Convolutional Generator for MNIST
    """
    def __init__(self, input_size=62, code_size=12, num_classes=784):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size+code_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(# input: 7 by 7, output: 14 by 14
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(# input: 14 by 14, output: 28 by 28
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            #nn.ReLU()
            nn.Tanh(),
        )
        
    def forward(self, z, c):
        z = z.view(z.size(0), -1)
        c = c.view(c.size(0), -1)
        noise = torch.cat((z, c), 1)
        x_ = self.layer1(noise)
        x_ = self.layer2(x_)
        x_ = x_.view(x_.size(0), 128, 7, 7)
        x_ = self.layer3(x_)
        x_ = self.layer4(x_)
        return x_


class Mine(nn.Module):
    """Convolutional Network for learning the MINE."""
    
    def __init__(self, sample_size=28*28, noise_size=12, output_size=1, hidden_size=128):
        super().__init__()
        self.layer1 = nn.Sequential(# 28 -> 14
            nn.Conv2d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.layer2 = nn.Sequential(# 14 -> 7
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
        )
        self.layer3 = nn.Sequential(#
            nn.Linear(128*7*7+noise_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
        )   

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 12),
        )
        self.flatten = nn.Flatten()
        self.cat = ConcatLayer()
                
    def forward(self, sample, noise):
        x_s = self.layer1(sample)
        x_s = self.layer2(x_s)
        flattened_sample = self.flatten(x_s)
        x = self.cat(flattened_sample, noise)
        x = self.layer3(x)
        x = self.fc(x)
        discrete = x[:10]
        cont = x[10:]
        
        return discrete, cont



class OldMine(nn.Module):
    """Old architecture for M."""
    
    def __init__(self, sample_size=28*28, noise_size=12, output_size=1, hidden_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.flatten = nn.Flatten()
        self.cat = ConcatLayer()
        self.layer1 = nn.Linear(64*4*4 + noise_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.activation1 = nn.LeakyReLU(0.1)
        self.layer3 = nn.Linear(512, 256)      
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 12)
        self.activation2=nn.ReLU()
        

        #self.layer1 = nn.Linear(sample_size + noise_size, 400)
        #self.layer2 = nn.Linear(400, 400)
        #self.layer3 = nn.Linear(400, 400)
        #self.layer4 = nn.Linear(400, 12)
        #self.activation = nn.ReLU()

                
    def forward(self, sample, noise):
        x_s = self.conv1(sample)
        x_s = self.activation2(x_s)
        x_s = self.conv2(x_s)
        x_s = self.activation2(x_s)
        x_s = self.conv3(x_s)
        x_s = self.activation2(x_s)
        flattened_sample = self.flatten(x_s)
        x = self.cat(flattened_sample, noise)
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation1(x)
        #x = self.layer3(x)
        #x = self.bn3(x)
        #x = self.activation1(x)
        x = self.layer4(x)
        #x = self.activation(x)
        discrete = x[:10]
        cont = x[10:]
        
        return discrete, cont