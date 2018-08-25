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

