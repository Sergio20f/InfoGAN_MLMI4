import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class W_Generator(nn.Module):
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
"""
class W_Generator(nn.Module):
    def __init__(self, input_size=62, code_size=62, num_classes=784):
        super(W_Generator, self).__init__()
        self.fc1 = nn.Linear(input_size+code_size, 1024) # +code_size
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 7*7*128)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()

        # Does not work with 1 layer for both nor for one for each
        #self.c_aug = nn.Linear(2, 52)
        self.c1_aug = nn.Linear(1, 26)
        self.c2_aug = nn.Linear(1, 26)
        
    def forward(self, z, c):
        z = z.view(z.size(0), -1)
        
        c = c.view(c.size(0), -1)
        c_discrete = c[:, :-2]
        c_cont = c[:, -2:]
        c_cont_1 = self.c1_aug(c_cont[:, :1])
        c_cont_2 = self.c2_aug(c_cont[:, 1:2])
        c_cont = torch.concat((c_cont_1, c_cont_2), 1)

        #c = c.view(c.size(0), -1)
        #c_discrete = c[:, :-2]
        #c_cont = self.c_aug(c[:, -2:])
        
        c = torch.concat((c_discrete, c_cont), 1)
        
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
    def __init__(self, in_channel=1, input_size=784, condition_size=10, num_classes=1):
        super(W_Discriminator, self).__init__()
        self.in_channel = in_channel
        self.input_size = input_size
        self.condition_size = 10
        self.num_classes = 1

        self.transform_layer = nn.Linear(input_size + condition_size, 784)
        self.conv_layer_1 = nn.Conv2d(self.in_channel, 512, 3, stride=2, padding=1, bias=False) # +1?
        self.batch_norm_1 = nn.BatchNorm2d(512)
        self.conv_layer_2 = nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(256)
        self.conv_layer_3 = nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc_layer_feat = nn.Linear(128, 1024)
        self.fc_layer = nn.Linear(128, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, c=None):
        # x: (N, 1, 28, 28), c: (N, 10)
        x, c = x.view(x.size(0), -1), c.float()
        v = torch.cat((x, c), 1)

        y_ = self.leaky_relu(self.transform_layer(v))
        y_ = y_.view(y_.shape[0], self.in_channel, 28, 28) # +1?
        y_ = self.leaky_relu(self.batch_norm_1(self.conv_layer_1(y_)))
        y_ = self.leaky_relu(self.batch_norm_2(self.conv_layer_2(y_)))
        y_ = self.leaky_relu(self.batch_norm_3(self.conv_layer_3(y_)))
        y_ = self.avg_pool(y_)
        y_ = y_.view(y_.size(0), -1)
        h = self.fc_layer_feat(y_)
        y_ = self.fc_layer(y_)

        return y_, h


class W_Qrator(nn.Module):
    """
    Regularization Network for increasing Mutual Information in InfoGAN. - SCALED

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
        c_discrete = c[:, :10] # Digit Label {0~9}
        c_mu = c[:, 10:12] # mu & var of Rotation & Thickness
        c_var = c[:, 12:14].exp() # mu & var of Rotation & Thickness
        return c_discrete, c_mu, c_var
    
"""
class W_Qrator(nn.Module):
    def __init__(self):
        super(W_Qrator, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.activation3 = nn.LeakyReLU(0.2, inplace=True)
        self.fc4 = nn.Linear(128, 14)
        
    def forward(self, x):
        # Seperate code
        c = self.fc1(x)
        c = self.bn1(c)
        c = self.activation1(c)
        c = self.fc2(c)
        c = self.bn2(c)
        c = self.activation2(c)
        c = self.fc3(c)
        c = self.bn3(c)
        c = self.activation3(c)
        c = self.fc4(c.detach())
        c_discrete = c[:, :10] # Digit Label {0~9}
        c_mu = c[:, 10:12] # mu & var of Rotation & Thickness
        c_var = c[:, 12:14].exp() # mu & var of Rotation & Thickness
        
        return c_discrete, c_mu, c_var
"""