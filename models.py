## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input image (1, 224, 224)
        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        # output tensor: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        I.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2.0))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # > relu or (ELUs) (32, 220, 220)
        # output tensor: (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.3)
        # output tensor: (64, 108, 108)
        
        self.conv2 = nn.Conv2d(32, 64, 3)
        I.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2.0))
        # ELUs(64, 108, 108)-> pool (64, 54, 54)-> dropout(0.4)(64, 54, 54)
        self.dropout2 = nn.Dropout(p=0.3)
        # output tensor: (128, 52, 52)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        I.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2.0))
        # ELUs(128, 52, 52)-> pool (128, 26, 26)-> dropout(0.4)(128, 26, 26)
        self.dropout3 = nn.Dropout(p=0.3)
        # output tensor: (256, 25, 25)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        I.xavier_uniform_(self.conv4.weight, gain=np.sqrt(2.0))
        self.dropout4 = nn.Dropout(p=0.4)
        # RLU(256, 25, 25)-> pool (256, 12, 12)-> dropout(0.4)(256, 12, 12)
        # faltten 256*12*12 -> 2000 -> Relu -> dropout (0.4)
        
        self.fc1 = nn.Linear(256*12*12, 2000)
        self.dropout5 = nn.Dropout(p=0.5)
        # Relu => dropout with 0.4
        
        self.fc2 = nn.Linear(2000, 2000)
        self.dropout6 = nn.Dropout(p=0.5)
        
        # finally, create 68 * 2 output keypoints
        self.fc3 = nn.Linear(2000, 68*2)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # output tensor: (32, 110, 110)
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        #bn = nn.BatchNorm2d(32, momentum=0.5).cuda()
        #x = bn(x)
        
        # output tensor: (64, 54, 54)
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        #bn = nn.BatchNorm2d(64, momentum=0.5).cuda()
        #x = bn(x)
        
        # output tensor: (128, 26, 26)
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        #bn = nn.BatchNorm2d(128, momentum=0.5).cuda()
        #x = bn(x)
        
        # output tensor: (256, 12, 12)
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        #bn = nn.BatchNorm2d(256, momentum=0.5).cuda()
        #x = bn(x)
        
        # prep for linear layer
        x = x.view(x.size(0), -1)
        # three linear layers with dropout in between
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        # final output
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class CNN1(nn.Module):

    def __init__(self):
        super(CNN1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input image (1, 224, 224)
        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        # output tensor: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        I.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2.0))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # > relu or (ELUs) (32, 220, 220)
        # output tensor: (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.1)
        # output tensor: (64, 108, 108)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        I.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2.0))
        # ELUs(64, 108, 108)-> pool (64, 54, 54)-> dropout(0.4)(64, 54, 54)
        self.dropout2 = nn.Dropout(p=0.2)
        # output tensor: (128, 52, 52)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        I.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2.0))
        # ELUs(128, 52, 52)-> pool (128, 26, 26)-> dropout(0.4)(128, 26, 26)
        self.dropout3 = nn.Dropout(p=0.3)
        # output tensor: (256, 25, 25)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        I.xavier_uniform_(self.conv4.weight, gain=np.sqrt(2.0))
        self.dropout4 = nn.Dropout(p=0.4)
        # RLU(256, 25, 25)-> pool (256, 12, 12)-> dropout(0.4)(256, 12, 12)
        
        # faltten 256*12*12 -> 2000 -> Relu -> dropout (0.4)
        self.fc1 = nn.Linear(256*12*12, 2000)
        self.dropout5 = nn.Dropout(p=0.4)
        # Relu => dropout with 0.4
        self.fc2 = nn.Linear(2000, 2000)
        self.dropout6 = nn.Dropout(p=0.4)
        # finally, create 68 * 2 output keypoints
        self.fc3 = nn.Linear(2000, 68*2)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # output tensor: (32, 110, 110)
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        # output tensor: (64, 54, 54)
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        # output tensor: (128, 26, 26)
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        # output tensor: (256, 12, 12)
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        # prep for linear layer
        x = x.view(x.size(0), -1)
        # three linear layers with dropout in between
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        # final output
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x