import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from utils import part4Plots, visualizeWeights

base_path = '/home/ercihan/Desktop/EE449/HW1/'

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    
if __name__ == "__main__":
    model_names = ['MLP_1', 'MLP_2', 'CNN_3', 'CNN_4', 'CNN_5']
    epoch_num = 15
    monte_carlo_num = 1
    results_list = []

    for model_name in model_names:
        results = {}

        relu_loss = []
        sig_loss = []
        relu_grad = []
        sig_grad = []

        # load the checkpoint dictionaries for all of the monte carlo runs for the given model
        for i in range(monte_carlo_num):
            i += 1
            path = base_path + 'Model_' + model_name + '_SGD_EpochNumber_' + str(epoch_num) + '/MonteCarlo' + str(i) + '/check_dic.pickle'
            path_sig = base_path + 'Model_' + model_name + '_sig_SGD_EpochNumber_' + str(epoch_num) + '/MonteCarlo' + str(i) + '/check_dic.pickle'
            temp = pickle_load(path)
            temp_sig = pickle_load(path_sig)

            print("loading pickles is done!")

            relu_loss.append(temp["train_loss"][epoch_num-1])
            sig_loss.append(temp_sig["train_loss"][epoch_num-1])

            relu_grad.append(temp["gradient_magnitude"][epoch_num-1].cpu().numpy())
            sig_grad.append(temp_sig["gradient_magnitude"][epoch_num-1].cpu().numpy())

        results['name'] = model_name
        results['relu_loss_curve'] = relu_loss
        results['sigmoid_loss_curve'] = sig_loss
        results['relu_grad_curve'] = relu_grad
        results['sigmoid_grad_curve'] = sig_grad

        results_list.append(results)
        pickle_dump(results_list, base_path + model_name + '/part4_' + model_name + '.pickle')

    part4Plots(results = results_list, save_dir=base_path+'results/', filename='result_part4', show_plot=True)
