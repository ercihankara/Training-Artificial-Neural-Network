import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
from torchvision.utils import make_grid

# written for the input images of size batchxNx32x32
# set the batch below!
batch = 50

# ANN structures and layers
class FullyConnected(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class MLP_1(torch.nn.Module):
    def __init__(self, input_size, channel, num_classes):
        super(MLP_1, self).__init__()
        self.channel = channel
        self.input_size_flatten = input_size*self.channel
        self.num_classes = num_classes

        # First fully connected layer
        self.fc1 = torch.nn.Linear(self.input_size_flatten, 32)
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Second fully connected layer
        self.fc2 = torch.nn.Linear(32, self.num_classes)

    def forward(self, x):
        #print("mlp1 input shape: ", str(x.shape))
        x = x.view(-1, self.input_size_flatten)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print("mlp1 output shape: ", str(x.shape))

        return x

class MLP_2(torch.nn.Module):
    def __init__(self, input_size, channel, num_classes):
        super(MLP_2, self).__init__()
        self.channel = channel
        self.input_size_flatten = input_size*self.channel
        self.num_classes = num_classes

        # First fully connected layer
        self.fc1 = torch.nn.Linear(self.input_size_flatten, 32)
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Fully connected layer
        self.fc2 = torch.nn.Linear(32, 64, bias = False)
        self.fc3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, x):
        #print("mlp2 input shape: ", str(x.shape))
        x = x.view(-1, self.input_size_flatten)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.relu1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print("mlp2 output shape: ", str(x.shape))

        return x

class CNN_3(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_3, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1, padding='valid')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*3*3, self.num_classes)

    def forward(self, x):
        #print("cnn3 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.relu1(x)
        #print("cnn3 before conv2 shape: ", str(x.shape))
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #print("cnn3 before conv3 shape: ", str(x.shape))
        x = self.conv3(x)
        x = self.maxpool3(x)
        #print("cnn3 before fc1 shape: ", str(x.shape))
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn3 output shape: ", str(x.shape))

        return x

class CNN_4(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_4, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU(inplace=True)

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fourth convolutional layer
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*4*4, self.num_classes)

    def forward(self, x):
        #print("cnn4 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn4 output shape: ", str(x.shape))

        return x

class CNN_5(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_5, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu1 = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu2 = torch.nn.ReLU(inplace=True)

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu3 = torch.nn.ReLU(inplace=True)

        # Fourth convolutional layer
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fifth convolutional layer
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu5 = torch.nn.ReLU(inplace=True)

        # Sixth convolutional layer
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.maxpool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(8*4*4, self.num_classes)

    def forward(self, x):
        #print("cnn5 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool6(x)
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn5 output shape: ", str(x.shape))

        return x

                    ##### PART 4 #####

class MLP_1_sig(torch.nn.Module):
    def __init__(self, input_size, channel, num_classes):
        super(MLP_1_sig, self).__init__()
        self.channel = channel
        self.input_size_flatten = input_size*self.channel
        self.num_classes = num_classes

        # First fully connected layer
        self.fc1 = torch.nn.Linear(self.input_size_flatten, 32)
        self.sig1 = torch.nn.Sigmoid()

        # Second fully connected layer
        self.fc2 = torch.nn.Linear(32, self.num_classes)

    def forward(self, x):
        #print("mlp1 input shape: ", str(x.shape))
        x = x.view(-1, self.input_size_flatten)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.sig1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print("mlp1 output shape: ", str(x.shape))

        return x

class MLP_2_sig(torch.nn.Module):
    def __init__(self, input_size, channel, num_classes):
        super(MLP_2_sig, self).__init__()
        self.channel = channel
        self.input_size_flatten = input_size*self.channel
        self.num_classes = num_classes

        # First fully connected layer
        self.fc1 = torch.nn.Linear(self.input_size_flatten, 32)
        self.sig1 = torch.nn.Sigmoid()

        # Resting fully connected layers
        self.fc2 = torch.nn.Linear(32, 64, bias = False)
        self.fc3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, x):
        #print("mlp2 input shape: ", str(x.shape))
        x = x.view(-1, self.input_size_flatten)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.sig1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        x = self.fc3(x)
        #print("mlp2 output shape: ", str(x.shape))

        return x

class CNN_3_sig(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_3_sig, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        # grayscaled input image, one channel
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig1 = torch.nn.Sigmoid()

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(5, 5), stride=1, padding='valid')
        self.sig2 = torch.nn.Sigmoid()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7), stride=1, padding='valid')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*3*3, self.num_classes)

    def forward(self, x):
        #print("cnn3 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.sig1(x)
        #print("cnn3 before conv2 shape: ", str(x.shape))
        x = self.conv2(x)
        x = self.sig2(x)
        x = self.maxpool2(x)
        #print("cnn3 before conv3 shape: ", str(x.shape))
        x = self.conv3(x)
        x = self.maxpool3(x)
        #print("cnn3 before fc1 shape: ", str(x.shape))
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn3 output shape: ", str(x.shape))

        return x

class CNN_4_sig(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_4_sig, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig1 = torch.nn.Sigmoid()

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig2 = torch.nn.Sigmoid()

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.sig3 = torch.nn.Sigmoid()
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fourth convolutional layer
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding='valid')
        self.sig4 = torch.nn.Sigmoid()
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*4*4, self.num_classes)

    def forward(self, x):
        #print("cnn4 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.sig1(x)
        x = self.conv2(x)
        x = self.sig2(x)
        x = self.conv3(x)
        x = self.sig3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.sig4(x)
        x = self.maxpool4(x)
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn4 output shape: ", str(x.shape))

        return x

class CNN_5_sig(torch.nn.Module):
    def __init__(self, channel, batch, num_classes):
        super(CNN_5_sig, self).__init__()
        self.batch = batch
        self.channel = channel
        self.num_classes = num_classes

        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels=self.channel, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig1 = torch.nn.Sigmoid()

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig2 = torch.nn.Sigmoid()

        # Third convolutional layer
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig3 = torch.nn.Sigmoid()

        # Fourth convolutional layer
        self.conv4 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig4 = torch.nn.Sigmoid()
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fifth convolutional layer
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig5 = torch.nn.Sigmoid()

        # Sixth convolutional layer
        self.conv6 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1, padding='valid')
        self.sig6 = torch.nn.Sigmoid()
        self.maxpool6 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        # Fully connected layer
        self.fc1 = torch.nn.Linear(8*4*4, self.num_classes)

    def forward(self, x):
        #print("cnn5 input shape: ", str(x.shape))
        x = self.conv1(x)
        x = self.sig1(x)
        x = self.conv2(x)
        x = self.sig2(x)
        x = self.conv3(x)
        x = self.sig3(x)
        x = self.conv4(x)
        x = self.sig4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.sig5(x)
        x = self.conv6(x)
        x = self.sig6(x)
        x = self.maxpool6(x)
        # make the input matrix a vector for the fully connected layer
        x = x.view(self.batch, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        #print("cnn5 output shape: ", str(x.shape))

        return x

if __name__ == "__main__":

    input = np.random.randint(0, 1, size =(2, 3, 32, 32)).astype(np.float32)
    input = torch.from_numpy(input)

    #ex2 = MLP_1(input_size = 32*32, channel = 1, num_classes = 10)
    #ex2.forward(input)
    ex = CNN_3(channel = 3, batch = 2)
    ex.forward(input)
