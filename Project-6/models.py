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
        
        # I decide to have 3 convolutional layers with following channel sizes
        ch1, ch2, ch3 = 32, 64, 128
        
        # corresponding filter sizes chosen are
        conv_params = {"ch1":5, "ch2":3, "ch3":3}
        # The input dimension is 224
        s = 224
        # Calculate the dimension for final max pooling layer
        for f in conv_params.values():
            s = (s-f+1)//2

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Conv Layer-1
        self.conv1 = nn.Conv2d(1, ch1, conv_params["ch1"])
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Layer-2
        self.conv2 = nn.Conv2d(ch1, ch2, conv_params["ch2"])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Layer-3
        self.conv3 = nn.Conv2d(ch2, ch3, conv_params["ch3"])
        self.pool3 = nn.MaxPool2d(2, 2)

        # Dense layers
        self.fc1 = nn.Linear(s**2 *ch3, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.output = nn.Linear(1000, 136) # Since we want to have 68 keypoints with 2 values each for x and y
        
        # Layers for avoiding the overfitting
        self.norm1 = nn.BatchNorm2d(ch1)
        self.norm2 = nn.BatchNorm2d(ch2)
        self.norm3 = nn.BatchNorm2d(ch3)
        
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.3)
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Conv Layers
        x = self.norm1(self.pool1(F.relu(self.conv1(x))))
        x = self.norm2(self.pool2(F.relu(self.conv2(x))))
        x = self.norm3(self.pool3(F.relu(self.conv3(x))))
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # Dense hidden layers
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        
        # output
        x = self.output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
