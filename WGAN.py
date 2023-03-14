import torch
import torch.nn as nn
import torch.nn.functional as F


class W_Generator(nn.Module):
    """
    Convolutional Generator for MNIST
    """
    def __init__(self, input_size=100, condition_size=10):
        super(W_Generator, self).__init__()
        
        self.input_size = input_size
        self.condition_size = condition_size
        
        self.fc1 = nn.Linear(input_size+condition_size, 4*4*512)
        self.conv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()
        
    def forward(self, x, c):
        x, c = x.view(x.size(0), -1), c.float()
        v = torch.cat((x, c), 1)
        
        y_ = self.fc1(v)
        y_ = y_.view(y_.size(0), 512, 4, 4)
        y_ = self.conv1(y_)
        y_ = self.bn1(y_)
        y_ = self.relu1(y_)
        y_ = self.conv2(y_)
        y_ = self.bn2(y_)
        y_ = self.relu2(y_)
        y_ = self.conv3(y_)
        y_ = self.tanh(y_)

        return y_



class W_Discriminator(nn.Module):
    """
    Convolutional Discriminator for MNIST

    """
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
        y_ = self.fc_layer(y_)

        return y_
