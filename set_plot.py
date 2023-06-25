import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle
from torchvision.utils import make_grid
from utils import part3Plots, visualizeWeights

base_path = '/content/drive/MyDrive/HW1/'

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
    
if __name__ == "__main__":
    model_names = ['MLP_1', 'MLP_2', 'CNN_3', 'CNN_4', 'CNN_5']
    epoch_num = 15
    monte_carlo_num = 10
    results_list = []

    for model_name in model_names:
        all_check_dic = []
        results = {}

        best_test_acc = 0
        best_test_acc_indx = 0

        avg_loss = []
        avg_train_acc = []
        avg_val_acc = []
        weights = []

        # load the checkpoint dictionaries for all of the monte carlo runs for the given model
        for i in range(monte_carlo_num):
            i += 1
            path = base_path + 'Model_' + model_name + '_EpochNumber_' + str(epoch_num) + '/MonteCarlo' + str(i) + '/check_dic.pickle'
            temp = pickle_load(path)
            all_check_dic.append(temp)
            print(temp["test_accuracy"])

            if temp["test_accuracy"] > best_test_acc:
                best_test_acc = temp["test_accuracy"]
                best_test_acc_indx = i

        print("the best test accuracy is obtained at the Monte Carlo number: ", str(best_test_acc_indx))
        print("loading pickles is done!")

        # run for each monte carlo
        for i in range(len(temp["train_loss"][epoch_num-1])): 
            total = 0
            avg = 0
            for j in range(monte_carlo_num):
                total += all_check_dic[j]["train_loss"][epoch_num-1][i]
                avg = total/monte_carlo_num
            avg_loss.append(avg)
        print(len(avg_loss))

        # run for each monte carlo
        for i in range(len(temp["train_accuracy"][epoch_num-1])): 
            total = 0
            avg = 0
            for j in range(monte_carlo_num):
                total += all_check_dic[j]["train_accuracy"][epoch_num-1][i]
                avg = total/monte_carlo_num
            avg_train_acc.append(avg)
        print(len(avg_train_acc))

            # run for each monte carlo
        for i in range(len(temp["validation_accuracy"][epoch_num-1])): 
            total = 0
            avg = 0
            for j in range(monte_carlo_num):
                total += all_check_dic[j]["validation_accuracy"][epoch_num-1][i]
                avg = total/monte_carlo_num
            avg_val_acc.append(avg)
        print(len(avg_val_acc))

        print("averaging of loss and accuracies is done!")

        results['name'] = model_name
        results['loss_curve'] = avg_loss
        results['train_acc_curve'] = avg_train_acc
        results['val_acc_curve'] = avg_val_acc
        results['test_acc'] = all_check_dic[best_test_acc_indx]["test_accuracy"]
        results['weights'] = all_check_dic[best_test_acc_indx]["first_layer_weights"]

        results_list.append(results)
        pickle_dump(results_list, base_path + model_name + '/part3_' + model_name + '.pickle')
        visualizeWeights(weights = results['weights'].cpu().numpy(), save_dir=base_path+'results/', filename='weights'+model_name)
  
    part3Plots(results = results_list, save_dir=base_path+'results/', filename='result_', show_plot=True)
