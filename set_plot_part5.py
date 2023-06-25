import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from utils import part5Plots, visualizeWeights

base_path = '/home/ercihan/Desktop/EE449/HW1/'

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    
if __name__ == "__main__":
    model_names = ['CNN_3']
    epoch_num = 20
    monte_carlo_num = 1

    for model_name in model_names:
        results = {}

        # load the checkpoint dictionaries for all of the monte carlo runs for the given model
        for i in range(monte_carlo_num):
            i += 1
            path_1 = base_path + 'Model_' + model_name + '_EpochNumber_' + str(epoch_num) + '_LearningRate_0.1/MonteCarlo' + str(i) + '/check_dic.pickle'
            path_01 = base_path + 'Model_' + model_name + '_EpochNumber_' + str(epoch_num) + '_LearningRate_0.01/MonteCarlo' + str(i) + '/check_dic.pickle'
            path_001 = base_path + 'Model_' + model_name + '_EpochNumber_' + str(epoch_num) + '_LearningRate_0.001/MonteCarlo' + str(i) + '/check_dic.pickle'
            temp_1 = pickle_load(path_1)
            temp_01 = pickle_load(path_01)
            temp_001 = pickle_load(path_001)

            print("loading pickles is done!")

            results['name'] = model_name
            results['loss_curve_1'] = temp_1['train_loss'][epoch_num-1]
            results['loss_curve_01'] = temp_01['train_loss'][epoch_num-1]
            results['loss_curve_001'] = temp_001['train_loss'][epoch_num-1]
            results['val_acc_curve_1'] = temp_1['validation_accuracy'][epoch_num-1]
            results['val_acc_curve_01'] = temp_01['validation_accuracy'][epoch_num-1]
            results['val_acc_curve_001'] = temp_001['validation_accuracy'][epoch_num-1]

        pickle_dump(results, base_path + model_name + '/part5_' + model_name + '.pickle')

    part5Plots(result = results, save_dir=base_path+'results/', filename='result_part5', show_plot=True)
