## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Input image size is (1, 164, 164) and thus the output will be (164-5+2(0) / 1) + 1 = 160 i.e (32, 160, 160)
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Output size = (32, 80, 80)
        self.pool1 = nn.MaxPool2d(2,2)
        
        # Input image size is (32, 80, 80) and thus the output will be (80-3+2(0) / 1) + 1 = 78 i.e (64, 78, 78)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Output size = (64, 39, 39)
        self.pool2 = nn.MaxPool2d(2,2)
        
        # Input image size is (64, 39, 39) and thus the output will be (39-2+2(0) / 1) + 1 = 38 i.e (128, 38, 38)
        self.conv3 = nn.Conv2d(64, 128, 2)
        # Output size = (128, 19, 19)
        self.pool3 = nn.MaxPool2d(2,2)
        
        # Input image size is (128, 19, 19) and thus the output will be (19-1+2(0) / 1) + 1 = 19 i.e (256, 19, 19)
        self.conv4 = nn.Conv2d(128, 256, 1)
        # Output size = (256, 9, 9)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256*9*9, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        #Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.35)
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
