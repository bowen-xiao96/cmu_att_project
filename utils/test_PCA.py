import matplotlib.pyplot as plt
import numpy as np
import torch
from PCA import my_PCA

b = np.load('/data2/simingy/pca/r_noise_50_0.npy')
c = np.load('/data2/simingy/pca/r_noise_50_1.npy')
d = np.load('/data2/simingy/pca/r_noise_50_2.npy')
e = np.load('/data2/simingy/pca/r_noise_50_3.npy')
f = np.load('/data2/simingy/pca/r_noise_50_4.npy')

total = np.zeros((250, 802816))
total[:50, :] = b.reshape(50, 802816)
total[50:100, :] = c.reshape(50, 802816)
total[100:150, :] = d.reshape(50, 802816)
total[150:200, :] = e.reshape(50, 802816)
total[200:250, :] = f.reshape(50, 802816)

total = torch.from_numpy(total)

new_total = my_PCA(total)

print(new_total.shape)

np.save('/data2/simingy/pca/r_noise_50.npy', new_total.numpy())

