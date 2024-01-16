import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        # convolution layers
        self.conv1 = nn.Conv2d(1, 40, 5)   
        self.conv2 = nn.Conv2d(40, 40, 5)  
        # fully connected layers
        self.fcl1 = nn.Linear(1 * 28 * 28, 100)
        self.fcl2 = nn.Linear(40 * 4 * 4, 100)
        self.fcl3 = nn.Linear(100, 100)
        self.fcl4 = nn.Linear(40 * 4 * 4, 1000)
        self.fcl5 = nn.Linear(1000, 1000)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)



    # flattening function
    def num_flat_features(self, x):
      '''
      calculates the flattened vector dimension size for input into the fully connected layers
      '''
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features


    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer.
        #
        fcl = X.view(-1, self.num_flat_features(X))
        fcl = F.sigmoid(self.fcl1(fcl))
        return  fcl


    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        fcl = F.max_pool2d(F.sigmoid(self.conv1(X)), 2)
        fcl = F.max_pool2d(F.sigmoid(self.conv2(fcl)), 2)

        fcl = fcl.view(-1, self.num_flat_features(fcl))
        fcl = F.sigmoid(self.fcl2(fcl))

        return  fcl

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        fcl = F.max_pool2d(F.relu(self.conv1(X)), 2)
        fcl = F.max_pool2d(F.relu(self.conv2(fcl)), 2)

        fcl = fcl.view(-1, self.num_flat_features(fcl))
        fcl = F.relu(self.fcl2(fcl))
        return  fcl

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        fcl = F.max_pool2d(F.relu(self.conv1(X)), 2)
        fcl = F.max_pool2d(F.relu(self.conv2(fcl)), 2)

        fcl = fcl.view(-1, self.num_flat_features(fcl))
        fcl = F.relu(self.fcl2(fcl))
        fcl = self.fcl3(fcl)
        return  fcl

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        fcl = F.max_pool2d(F.relu(self.conv1(X)), 2)
        fcl = F.max_pool2d(F.relu(self.conv2(fcl)), 2)

        fcl = fcl.view(-1, self.num_flat_features(fcl))
        fcl = F.relu(self.fcl4(fcl))
        fcl = self.fcl5(fcl)
        return  fcl

