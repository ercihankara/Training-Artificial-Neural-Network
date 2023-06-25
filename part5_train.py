import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import gc
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import NN_arch as na

from utils_ad import download_data, train_v3, acc_calc, pickle_dump

curr_dir = os.getcwd()

if __name__ == "__main__":
    # CUDA for PyTorch
    gc.collect()
    torch.cuda.empty_cache()

    # check the GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(798)

    monte_carlo_seeds_single = [1]
    learning_rates = [0.1, 0.01, 0.001]

    model_name = 'CNN_3'

    batch_size = 50
    epoch_num = 20

    ###### download and split the dataset ######

    # set download option as False if the dataset is already donwloaded
    train_data, test_data = download_data()

    #split the training dataset to create the validation dataset
    train_set, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    ###### ... ######

    for i, monte_carlo_seed in enumerate(monte_carlo_seeds_single):
        for lr in enumerate(learning_rates):
            print("Monte Carlo: ", str(monte_carlo_seed))
            print("Learning Rate: ", str(lr))
            torch.manual_seed(monte_carlo_seed)

            # initialize checkpoint dictionary
            check_dic = {}

            # initialize the dataloaders
            train_generator = DataLoader(train_set, batch_size = batch_size, shuffle = True)
            val_generator = DataLoader(val_data, batch_size = batch_size, shuffle = True)
            test_generator = DataLoader(test_data, batch_size = batch_size, shuffle = False)

            print("train size: ", str(len(train_generator.dataset)))
            print("val size: ", str(len(val_generator.dataset)))

            # initialize the model
            model = na.CNN_3(1, batch_size, 10)
            model.cuda()

            # set the optimizer
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
            optimizer.zero_grad()

            # create loss function using cross entropy loss
            loss = torch.nn.CrossEntropyLoss()

            # create the save directory
            save_dir = curr_dir + f'/Model_{model_name}_EpochNumber_{epoch_num}_LearningRate_' + str(lr) + f'/MonteCarlo{monte_carlo_seed}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # transfer your model to train mode
            train_v3(model = model, optimizer = optimizer, loss_func = loss,
                        train_loader = train_generator, val_loader = val_generator,
                        epoch_num = epoch_num, model_name = model_name, check_dic = check_dic, device = device)

            # save the model with other parameters at the end of the training
            pickle_dump(check_dic, save_dir + '/check_dic.pickle')
