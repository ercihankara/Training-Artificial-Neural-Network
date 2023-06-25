import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
import gc
from tqdm import tqdm
import pickle
import random
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision.utils import make_grid
import torch.utils.data

# download and load the dataset
def download_data():

    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # normalize the images between -1 and 1
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    torchvision.transforms.Grayscale()
    ])
    # training set
    train_data = torchvision.datasets.CIFAR10('./data', train = True, download = True,
    transform = transform)
    # test set
    test_data = torchvision.datasets.CIFAR10('./data', train = False,
    transform = transform)

    return train_data, test_data

def pickle_dump(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def pickle_load(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def acc_calc(model, loader, device):
    # disable gradient tracking
    with torch.no_grad():
      # calculate the training accuracy
      correct = 0
      total = 0
      for inputs, labels in loader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = model.forward(inputs)
          _, predicted = outputs.max(1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      accuracy = 100*correct/total

    return accuracy

# FOR PART 3
def train(model, optimizer, loss_func,
                   train_loader, val_loader, epoch_num, model_name, check_dic, device = torch.device('cpu')):
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()
    print('DEVICE: ', device)

    train_acc = []
    val_acc = []
    loss_total = []

    loss_total_complete = []
    train_acc_complete = []
    val_acc_complete = []

    print("Train size: " + str(len(train_loader.dataset)))
    print("\n")

    # training loop
    for epoch in range(epoch_num):
        losses = 0.0
        print("epoch: ", str(epoch))
        
        for i, (train_in, train_target) in tqdm(enumerate(train_loader)):
            # training mode
            model.train()

            # send data to the device
            train_in = train_in.to(device)
            train_target = train_target.to(device)

            # forward propagation
            estimation = model.forward(train_in)

            # back propagation
            # calculate loss
            loss = loss_func(estimation, train_target.to(device))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # optimization
            optimizer.step()

            if i % 10 == 0:
                model.eval()

                # training and validation accuracy
                _, training_prediction = estimation.max(1)
                training_size = train_target.size(0)
                training_corr = training_prediction.eq(train_target).sum().item()
                training_acc = (training_corr / training_size) * 100

                # loss integration
                losses = loss.item()
                print("loss: ", str(losses))

                loss_total.append(losses)
                train_acc.append(training_acc)
                print("train acc: ", str(training_acc))
                val_acc.append(acc_calc(model, val_loader, device))
                print("val acc: ", str(acc_calc(model, val_loader, device)))

        # final accuracy and loss append for each epoch, cumulative
        loss_total_complete.append(loss_total)
        train_acc_complete.append(train_acc)
        val_acc_complete.append(val_acc)

    print("training accuracy: ", str(train_acc))

    # if mlp
    if "MLP" in model_name:
        first_layer_weights = model.fc1.weight.data
    # if cnn
    else:
        first_layer_weights = model.conv1.weight.data

    check_dic["name"] = model_name
    check_dic["train_loss"] = loss_total_complete
    check_dic["train_accuracy"] = train_acc_complete
    check_dic["validation_accuracy"] = val_acc_complete
    check_dic["first_layer_weights"] = first_layer_weights

    return

# FOR PART 4
def train_v2(model, optimizer, loss_func,
                   train_loader, epoch_num, model_name, check_dic, device = torch.device('cpu')):
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()
    print('DEVICE: ', device)

    loss_total = []
    grad_magnitude = []

    loss_total_complete = []
    grad_magnitude_complete = []

    print("Train size: " + str(len(train_loader.dataset)))
    print("\n")

    # training loop
    for epoch in range(epoch_num):
        losses = 0.0
        print("epoch: ", str(epoch))
        
        for i, (train_in, train_target) in tqdm(enumerate(train_loader)):
            # training mode
            model.train()

            # send data to the device
            train_in = train_in.to(device)
            train_target = train_target.to(device)

            # forward propagation
            estimation = model.forward(train_in)

            # back propagation
            # calculate loss
            loss = loss_func(estimation, train_target.to(device))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # optimization
            optimizer.step()

            if i % 10 == 0:
                model.eval()

                # loss integration
                losses = loss.item()
                print("loss: ", str(losses))

                loss_total.append(losses)
                print("train loss: ", str(losses))

                # if mlp
                if "MLP" in model_name:
                    grad_mag = torch.norm(model.fc1.weight.grad)
                # if cnn
                else:
                    grad_mag = torch.norm(model.conv1.weight.grad)
                grad_magnitude.append(grad_mag)
                print("gradient: ", str(grad_mag))

        # final accuracy and loss append for each epoch, cumulative
        loss_total_complete.append(loss_total)
        grad_magnitude_complete.append(grad_magnitude)

    check_dic["name"] = model_name
    check_dic["train_loss"] = loss_total_complete
    check_dic["gradient_magnitude"] = grad_magnitude_complete

    return

# FOR PART 5
def train_v3(model, optimizer, loss_func,
                   train_loader, val_loader, epoch_num, model_name, check_dic, device = torch.device('cpu')):
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()
    print('DEVICE: ', device)

    val_acc = []
    loss_total = []

    loss_total_complete = []
    val_acc_complete = []

    print("Train size: " + str(len(train_loader.dataset)))
    print("\n")

    # training loop
    for epoch in range(epoch_num):
        losses = 0.0
        print("epoch: ", str(epoch))

        for i, (train_in, train_target) in tqdm(enumerate(train_loader)):
            # training mode
            model.train()

            # send data to the device
            train_in = train_in.to(device)
            train_target = train_target.to(device)

            # forward propagation
            estimation = model.forward(train_in)

            # back propagation
            # calculate loss
            loss = loss_func(estimation, train_target.to(device))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # optimization
            optimizer.step()

            if i % 10 == 0:
                model.eval()

                # loss integration
                losses = loss.item()
                print("loss: ", str(losses))

                loss_total.append(losses)

                # validation accuracy
                val_acc.append(acc_calc(model, val_loader, device))
                print("val acc: ", str(acc_calc(model, val_loader, device)))

        # final accuracy and loss append for each epoch, cumulative
        loss_total_complete.append(loss_total)
        val_acc_complete.append(val_acc)

    check_dic["name"] = model_name
    check_dic["train_loss"] = loss_total_complete
    check_dic["validation_accuracy"] = val_acc_complete

    return

# FOR PART 5, p2
def train_v4(model, optimizer, loss_func,
                   train_loader, val_loader, epoch_num, model_name, check_dic, device = torch.device('cpu')):
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.empty_cache()
    print('DEVICE: ', device)

    val_acc = []
    val_acc_complete = []

    holder0 = True
    holder1 = True

    print("Train size: " + str(len(train_loader.dataset)))
    print("\n")

    # training loop
    for epoch in range(epoch_num):
        losses = 0.0
        print("epoch: ", str(epoch))

        # increase in accuracy stops about 300th iteration (about 3th epoch), decrease lr value, update optimizer!
        if((epoch==3) and (holder0)):
            optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.00)
            holder0 = False
            print("Learning Rate: ", 0.01)

        # increase in accuracy stops about 900th iteration (about 10th epoch), decrease lr value again, update optimizer!
        if((epoch==10) and (holder1)):
            optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.00)
            holder1 = False
            print("Learning Rate: ", 0.001)

        for i, (train_in, train_target) in tqdm(enumerate(train_loader)):
            # training mode
            model.train()

            # send data to the device
            train_in = train_in.to(device)
            train_target = train_target.to(device)

            # forward propagation
            estimation = model.forward(train_in)

            # back propagation
            # calculate loss
            loss = loss_func(estimation, train_target.to(device))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # optimization
            optimizer.step()

            if i % 10 == 0:
                model.eval()

                # validation accuracy
                val_acc.append(acc_calc(model, val_loader, device))
                print("val acc: ", str(acc_calc(model, val_loader, device)))

        # final accuracy append for each epoch, cumulative
        val_acc_complete.append(val_acc)

    check_dic["name"] = model_name
    check_dic["validation_accuracy"] = val_acc_complete

    return

# custom conv2d function; convolution without flipping of kernel
# no padding, one stride
def my_conv2d(input, kernel):

    # get the input and kernel dimensions
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, input_channels, filter_height, filter_width = kernel.shape

    # calculate the output dimensions
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1

    # create the output tensor
    output = np.zeros((batch_size, output_channels, output_height, output_width))

    # perform convolution, 4D input & kernel
    for batch in range(batch_size):
        for out_ch in range(output_channels):
            for in_ch in range(input_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        output[batch, out_ch, i, j] += np.sum(
                            input[batch, in_ch, i:i+filter_height, j:j+filter_width] * kernel[out_ch, in_ch, :, :]
                        )

    return output

# utility function to create validation accuracy plot for part 5, p2
def part5Plots_val_acc(result, save_dir='', filename='', show_plot=True):
    """plots multiple performance curves from multiple training results and
    saves the resultant plot as a png image

    Arguments:
    ----------

    result: dictionary object, each corresponds to
    the result of a training and should have the following key-value
    items:

        'name': string, indicating the user-defined name of the training

        'validation_accuracy': list of floats, indicating the val acc at each step

    save_dir: string, indicating the path to directory where the plot image is to be saved

    filename: string, indicating the name of the image file. Note that .png will be automatically
    appended to the filename.

    show_plot: bool, whether the figure is to be shown

    Example:
    --------

    visualizing the results of the training

    # assume the '*_value's are known

    >>> result = {'name': name_value, 'validation_accuracy': validation_accuracy_value}

    >>> part4Plots(result, save_dir=r'some\location\to\save', filename='part4Plots')

    """

    if isinstance(result, (list, tuple)):
        result = result[0]

    color_list_up = ['#ff00ff', '#ff00ff', '#ff00ff', '#ff00ff', '#ff00ff', '#ff00ff', '#ff00ff']
    style_list = ['-', '--']

    num_curves = 3

    plot_args = [{'c': color_list_up[k],
                  'linestyle': style_list[0],
                  'linewidth': 2} for k in range(num_curves)]

    key_suffixes = ['1', '01', '001']

    font_size = 18

    fig, axes = plt.subplots(1, figsize=(16, 12))

    fig.suptitle('training of <%s> with different learning rates' % result['name'],
                 fontsize=font_size, y=0.025)

    # training loss and validation accuracy
    axes.set_title('validation_accuracies with all three learning rates', loc='right', fontsize=font_size)
    for key_suffix, plot_args in zip(key_suffixes, plot_args):
        acc_curve = result['validation_accuracy']
        label = 'lr=0.%s' % key_suffix

        axes.plot(np.arange(1, len(acc_curve) + 1),
                     acc_curve, label=label, **plot_args)
        axes.set_xlabel(xlabel='step', fontsize=font_size)
        axes.set_ylabel(ylabel='accuracy', fontsize=font_size)
        axes.tick_params(labelsize=12)

    # global legend
    lines = axes.get_lines()
    fig.legend(labels=[line._label for line in lines],
               ncol=3, loc="upper center", fontsize=font_size,
               handles=lines)

    if show_plot:
        plt.show()

    fig.savefig(os.path.join(save_dir, filename + '.png'))
