import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import torch
import torchvision
from torch.utils.data import DataLoader

from utils import part2Plots
from utils_ad import download_data, my_conv2d
from NN_arch import FullyConnected

curr_dir = os.getcwd()

if __name__ == "__main__":

    """# download the dataset
    train_data, test_data = download_data()

    # define the dataloaders
    train_generator = DataLoader(train_data, batch_size = 96, shuffle = True)
    test_generator = DataLoader(test_data, batch_size = 96, shuffle = False)

    # initialize the model
    model_mlp = FullyConnected(1024,128,10)

    # get the parameters 1024x128 layer as numpy array
    params_784x128 = model_mlp.fc1.weight.data.numpy()

    # create loss: use cross entropy loss)
    loss = torch.nn.CrossEntropyLoss()
    # create optimizer
    optimizer = torch.optim.SGD(model_mlp.parameters(), lr = 0.01, momentum = 0.0)
    # transfer your model to train mode
    model_mlp.train()
    # transfer your model to eval mode
    model_mlp.eval()"""

    # download the input and kernel files

    # input shape: [batch size, input_channels, input_height, input_width]
    input = np.load('samples_0.npy')
    # input shape: [output_channels, input_channels, filter_height, filter width]
    kernel = np.load('kernel.npy')
    im1 = plt.imshow(np.reshape(input[0], (28, 28)))
    plt.show()

    print(input.shape)
    print(kernel.shape)

    # custom conv2d function
    out = my_conv2d(input, kernel)
    print(out.shape)
    part2Plots(out, save_dir = curr_dir, filename = 'part2_output_scaled')
