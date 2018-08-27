import numpy as np
import math
import matplotlib.pyplot as plt
import os

def show_status(expId, data_path='/data2/simingy/model/'):
    data_file = os.path.join(data_path, expId, 'training_data.npz')
    data_file = np.load(data_file)

    acc = data_file['acc']
    lr = data_file['lr']

    plt.figure(figsize=(10,5)) 
    plt.subplot(1, 2, 1)
    best_acc = 0
    for i in range(acc.shape[0]):
        if best_acc < acc[i]:
            best_acc = acc[i]

        if acc[i] == 0:
            right_boundary = i
            break
    print("best top1:", best_acc[0])
    x_list = np.arange(0, right_boundary, 1)
    y_list = np.zeros((right_boundary))
    for i in range(right_boundary):
        y_list[i] = acc[i]

    plt.plot(x_list, y_list)
    plt.title('test accuracy')
    
    plt.subplot(1, 2, 2)


    x_list = np.arange(0, right_boundary, 1)
    y_list = np.zeros((right_boundary))
    for i in range(right_boundary):
        y_list[i] = lr[i]

    plt.plot(x_list, y_list)
    plt.title('learning rate')


def show_weights(expId, data_path='/data2/simingy/model'):
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots()
    ax.set_color_cycle(['red', 'orange', 'yellow', 'green', 'blue', 'black'])
    plt.xlabel('number of neuron')
    plt.ylabel('neuron response')
    
    for i in range(6):
        data_file = os.path.join(data_path, expId, 'gate_' + str(i) + '.npz')
        data_file = np.load(data_file)
        gate_weights = data_file['gate']
        gate_weights = np.abs(gate_weights)

        gate_max = np.max(gate_weights)
        x_axis = []
        y_axis = []
        for j in range(10):
            x_axis.append((gate_weights > gate_max * (9-j) * 0.1).sum())
            y_axis.append(gate_max * (9-j) * 0.1)

        plt.plot(x_axis, y_axis, label=str(i+1)+'epoch')
    plt.legend(loc = 'best')


def show_distribution(expId, data_path='/data2/simingy/model'):
    for i in range(6):
        plt.figure(figsize=(5, 5))
        data_file = os.path.join(data_path, expId, 'gate_' + str(i) + '.npz')
        data_file = np.load(data_file)
        gate_weights = data_file['gate']
        flatten_weights = gate_weights.flatten()
        plt.hist(flatten_weights, 50)
        
        
def show_seperate_weights(expId, data_path='/data2/simingy/model'):
    
    ratio_list = 5 * [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    hl_list = []
    for i in range(6):
        plt.figure(figsize=(5,5))
        data_file = os.path.join(data_path, expId, 'gate_' + str(i) + '.npz')
        data_file = np.load(data_file)
        gate_weights = data_file['gate']
        gate_weights = np.abs(gate_weights)
        high_weights = gate_weights[:, :512, :, :]
        low_weights = gate_weights[:, 512:, :, :]
        high_weights_max = np.max(high_weights)
        low_weights_max = np.max(low_weights)
        mmax = max(high_weights_max, low_weights_max)
        high_firing = (high_weights > mmax * ratio_list[i]).sum() / 2
        low_firing = (low_weights > mmax * ratio_list[i]).sum()
        x_axis = []
        x_axis.append(high_firing)
        x_axis.append(low_firing)
        hl_list.append(float(high_firing) / low_firing)
        plt.bar(left=[0, 1], height=x_axis)
    
    time_list = [1, 2, 3, 4, 5, 6]
    plt.figure(figsize=(5,5))
    plt.plot(time_list, hl_list)
    plt.xlabel('epoch')
    plt.ylabel('high/low')
        
def show_spatial_weights(expId, data_path='/data2/simingy/model'):
    
    ratio_list = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for i in range(6):
        plt.figure(figsize=(5,5))
        data_file = os.path.join(data_path, expId, 'gate_' + str(i) + '.npz')
        data_file = np.load(data_file)
        gate_weights = data_file['gate']
        gate_weights = np.abs(gate_weights)
        
        # center part
        high_weights = gate_weights[:, :512, :, :]
        high_center_weights = gate_weights[:, :512, 1:3, 1:3]
        high_center_ave = high_center_weights.sum() / np.product(high_center_weights.shape)
        high_marginal_ave = (high_weights.sum() - high_center_weights.sum()) / (np.product(high_weights.shape) - np.product(high_center_weights.shape))
        plt.bar(left=[0, 1], height=[high_center_ave, high_marginal_ave])
        
        plt.figure(figsize=(5,5))
        low_weights = gate_weights[:, 512:, :, :]
        low_center_weights = gate_weights[:, 512:, 1:3, 1:3]
        low_center_ave = low_center_weights.sum() / np.product(low_center_weights.shape)
        low_marginal_ave = (low_weights.sum() - low_center_weights.sum()) / (np.product(low_weights.shape) - np.product(low_center_weights.shape))
        plt.bar(left=[0, 1], height=[low_center_ave, low_marginal_ave])
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
