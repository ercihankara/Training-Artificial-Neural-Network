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
    part2Plots(out, save_dir = curr_dir, filename = 'part2_output')
