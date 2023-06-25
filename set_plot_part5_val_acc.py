import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from utils_ad import part5Plots_val_acc

base_path = '/home/ercihan/Desktop/EE449/HW1/'

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

if __name__ == "__main__":
    model_names = ['CNN_3']
    epoch_num = 30
    monte_carlo_num = 1

    for model_name in model_names:
        results = {}

        # load the checkpoint dictionaries for all of the monte carlo runs for the given model
        for i in range(monte_carlo_num):
            i += 1
            path = base_path + 'Model_' + model_name + '_EpochNumber_' + str(epoch_num) + '_decreasing/MonteCarlo' + str(i) + '/check_dic.pickle'
            temp = pickle_load(path)

            print("loading pickles is done!")

            results['name'] = model_name
            results['validation_accuracy'] = temp['validation_accuracy'][epoch_num-1]

            print("test accuracy: ", str(temp['test_accuracy']))

        #pickle_dump(results, base_path + '/part5_dec_p1_' + model_name + '.pickle')

    part5Plots_val_acc(result = results, save_dir=base_path+'results/', filename='result_part5_dec', show_plot=True)
