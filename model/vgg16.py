import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A
import numpy as np
sys.path.insert(0, '/home/simingy/cmu_att_project/utils/')
from PCA import my_PCA


network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

j = 0
dprime_parameters = np.zeros((38, 1000, 2))

class VGG16(nn.Module):
    def __init__(self, network_cfg, num_class, last_dim=7, dropout=0.5):
        super(VGG16, self).__init__()

        # backbone network
        self.backbone = nn.ModuleList()
        input_dim = 3
        for v in network_cfg:
            if v == 'M':
                self.backbone.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.backbone.append(nn.Conv2d(input_dim, v, kernel_size=3, padding=1))
                self.backbone.append(nn.ReLU(inplace=True))
                input_dim = v

        # we use fully connected layer as classifier
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512 * last_dim * last_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_class)
        )
        '''
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(512 * last_dim * last_dim, 4096))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Dropout(p=dropout))
        self.classifier.append(nn.Linear(4096, 4096))
        self.classifier.append(nn.ReLU(inplace=True))
        self.classifier.append(nn.Dropout(p=dropout))
        self.classifier.append(nn.Linear(4096, num_class))

    def forward(self, x):
        global j
        global dprime_parameters
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            temp_list = [8, 9, 10, 11, 13, 15, 18] 
            temp_dic = {'8': 0, '9': 1, '10': 2, '11': 3, '13': 4, '15': 5, '16': 6, '18': 7, '20': 8, '22': 9, '23': 10, '25': 11, '27': 12, '29': 13}
            #if i in temp_list:
               #a = x.view(x.size(0), -1)
               #a = my_PCA(a)
               #plt.figure()
               #plt.scatter(a[:,0], a[:,1])
               #plt.savefig('/home/simingy/test.png')
               #np.save('/data2/simingy/pca/noise_50.npy', a.data.cpu().numpy())
                
            if j < 1000:
                #dprime_parameters[temp_dic[str(i)]][j][0] = torch.mean(x).data.cpu().numpy()
                #dprime_parameters[temp_dic[str(i)]][j][1] = torch.std(x).data.cpu().numpy()
                dprime_parameters[i][j][0] = torch.mean(x).data.cpu().numpy()
                dprime_parameters[i][j][1] = torch.std(x).data.cpu().numpy()

            #if i == 30:
            #    j = j + 1

            if j == 1000:
                np.save('/data2/simingy/data/dprime/vgg16_noise50/vgg16_noise50_layer{}.npy'.format(str(i)), dprime_parameters[i])
            
        x = x.view(x.size(0), -1)

        for i in range(7):
            x = self.classifier[i](x)
            if j < 1000:
                dprime_parameters[i+31][j][0] = torch.mean(x).data.cpu().numpy()
                dprime_parameters[i+31][j][1] = torch.std(x).data.cpu().numpy()
            if j == 1000:
                np.save('/data2/simingy/data/dprime/vgg16_noise50/vgg16_noise50_layer{}.npy'.format(str(i+31)), dprime_parameters[i+31])
        j = j + 1

        return x
