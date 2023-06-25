import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import gc
from tqdm import tqdm
import pickle
from statistics import mean
import random
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision.utils import make_grid
import torch.utils.data

base_path = 'D:/Ercihan/others/HW1/'
#Model_MLP_1_EpochNumber_15/MonteCarlo1/check_dic.pickle

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    
if __name__ == "__main__":
    path = 'D:/Ercihan/others/HW1/part3_MLP1_1.pkl'
    x = pickle_load(path)
    print(len(x["loss curve"]))