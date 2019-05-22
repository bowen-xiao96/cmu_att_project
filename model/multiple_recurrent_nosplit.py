import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A

sys.path.insert(0, '/home/simingy/cmu_att_project')
from model.gate import *
sys.path.insert(0, '/home/simingy/cmu_att_project/utils/')
from PCA import my_PCA

network_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

k = 0
dprime_parameters = np.zeros((38, 1000, 2))

class MultipleRecurrentModel(nn.Module):
    def __init__(self, network_cfg, connections, unroll_count, num_class,
                 gating_module=GatingModule, last_dim=7, dropout=0.5):
        super(MultipleRecurrentModel, self).__init__()

        # connections: (high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor)
        # high_layer, low_layer: layer index
        # scale_factor: 2**n

        # ** data flows from high_layer to low_layer (recurrent) **
        # ** currently, there should not be two connections pointing to a same low_layer **

        self.unroll_count = unroll_count

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

        # set up recurrent connections
        self.end_layers = set([c[0] for c in connections])
        self.max_end_layer = max(self.end_layers)
        print("max_end_layer:", self.max_end_layer)

        self.layer_count = len(self.backbone)
        print("layer_count:", self.layer_count)
        self.gating = nn.ModuleList()
        self.point_to = [list() for _ in self.backbone]

        # sort connections according to low_layer
        # since there should not be two connections pointing to a same low_layer
        # the modules in classifiers and gating modules are also in this order (low_layer)
        # connections = sorted(connections, key=lambda c: c[1])
        
        self.loop_num = 0
        for high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor in connections:
            # can modify the settings below
            # kernel_size, stride, padding, output_padding
            self.loop_num += 1
            if scale_factor == 1:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 1, 1, 0), 1, 3, dropout=0.5)
            elif scale_factor == 2:
                gating = gating_module(low_layer_dim, high_layer_dim, (3, 2, 1, 1), 1, 3, dropout=0.5)
            else:
                # TODO: long connection across two blocks
                raise NotImplementedError
            print("gating:", gating)
            self.gating.append(gating)
            self.point_to[high_layer].append(low_layer)

    def forward(self, x):
        # each unrolling step, the model makes prediction at the final classifier

        # the input to each layer (not output)
        global k
        global dprime_parameters

        buf = [list() for _ in self.backbone]

        # the intermediate predictions of the network
        # the output should be a batch_size * unroll_count * num_class Tensor
        intermediate_pred = list()

        # layers before recurrent start
        for i in range(self.max_end_layer):
            # do not actually include the starting layer
            x = self.backbone[i](x)
            buf[i].append(x)
            temp_list = [8, 9, 10, 11, 13, 15, 16, 18] 
            temp_dic = {'8': 0, '9': 1, '10': 2, '11': 3, '13': 4, '15': 5, '16': 6, '18': 7, '20': 8, '22': 9, '23': 10, '25': 11, '27': 12, '29': 13}
            
            #if i in temp_list:
               #print(x.shape)
               #a = x.view(x.size(0), -1)
               #a = my_PCA(a)

               #np.save('/data2/simingy/pca/r_noise_50_4.npy', a.data.cpu().numpy())
               #print(x.shape)
            if i < 8:
                if k < 1000:
                    dprime_parameters[i][k][0] = torch.mean(x).data.cpu().numpy()
                    dprime_parameters[i][k][1] = torch.std(x).data.cpu().numpy()
            #if i == 18:
            #       k = k + 1
               #print(k)
                if k == 1000:
                    np.save('/data2/simingy/data/dprime/recurrent_l3_u4_noise50/recurrent_l3_u4_noise50_layer{}.npy'.format(str(i)), dprime_parameters[i])
            
            
            for point_to in self.point_to[i]:
                buf[point_to].append(x)

        buf[self.point_to[self.max_end_layer][0]].append(x)
        #print("Done!")

        # do recurrent stuff
        for i in range(self.unroll_count):
            if i == 0:
                # the first unrolling step
                # just straightforward forward computation
                for j in range(self.max_end_layer, self.layer_count):
                    x = self.backbone[j](x)

                    # put it into the buffer if there is feedback connection
                    #for point_to in self.point_to[j]:
                    #    buf[point_to].append(x)
                    '''
                    if k < 1000:
                        dprime_parameters[j][k][0] = torch.mean(x).data.cpu().numpy()
                        dprime_parameters[j][k][1] = torch.std(x).data.cpu().numpy()
                    if k == 1000:
                        np.save('/data2/simingy/data/dprime/recurrent_noise50/recurrent_noise50_layer{}.npy'.format(str(j)), dprime_parameters[j])
                    '''
                   

                # make final predictions
                x = x.view(x.size(0), -1)
                pred = x
                for j in range(7):
                    pred = self.classifier[j](pred)
                    '''
                    if k < 1000:
                        dprime_parameters[j+self.layer_count][k][0] = torch.mean(pred).data.cpu().numpy()
                        dprime_parameters[j+self.layer_count][k][1] = torch.std(pred).data.cpu().numpy()
                    if k == 1000:
                        np.save('/data2/simingy/data/dprime/recurrent_noise50/recurrent_noise50_layer{}.npy'.format(str(j+self.layer_count)), dprime_parameters[j+self.layer_count])
                    '''
                #k = k + 1           
                intermediate_pred.append(pred)
                #print("Done2!")

            else:
                # we need to apply recurrent to the input of each recurrent block
                # and make intermediate predictions using the prediction module

                # record which gating_models to use
 
                pos = 0
                pointer = self.point_to[self.max_end_layer][0]
                low, high = buf[pointer][-2:]
                x = self.gating[pos](low, high)
                buf[pointer].append(x)
                pos += 1

                for j in range(self.loop_num - 1):
                    pointer += 1
                    x = self.backbone[pointer](x)
                    pointer += 1
                    x = self.backbone[pointer](x)
                    pointer = self.point_to[pointer][0]
                    low, high = buf[pointer][-2:]
                    x = self.gating[pos](low, high)
                    buf[pointer].append(x)
                    pos += 1
                
                if i == self.unroll_count - 1:
                    if k < 1000: 
                        dprime_parameters[pointer][k][0] = torch.mean(x).data.cpu().numpy()
                        dprime_parameters[pointer][k][1] = torch.std(x).data.cpu().numpy()
                
                pointer += 1
                #print("Done3!")

                for j in range(pointer, self.layer_count):
                    
                    x = self.backbone[j](x)

                    # put it into the buffer if there is feedback connection
                    for point_to in self.point_to[j]:
                        buf[point_to].append(x)
                    
                    #if j in temp_list:
                        #print(x.shape)
                        #a = x.view(x.size(0), -1)
                        #a = my_PCA(a)

                        #np.save('/data2/simingy/pca/r_noise_50_4.npy', a.data.cpu().numpy())
                        #print(x.shape)
                    if i == self.unroll_count - 1:
                        if k < 1000:
                            dprime_parameters[j][k][0] = torch.mean(x).data.cpu().numpy()
                            dprime_parameters[j][k][1] = torch.std(x).data.cpu().numpy()

                        #if j == 18:
                        #    k = k + 1
                            #print(k)
                        if k == 1000:
                            np.save('/data2/simingy/data/dprime/recurrent_l3_u4_noise50/recurrent_l3_u4_noise50_layer{}.npy'.format(str(8)), dprime_parameters[8])
                            np.save('/data2/simingy/data/dprime/recurrent_l3_u4_noise50/recurrent_l3_u4_noise50_layer{}.npy'.format(str(j)), dprime_parameters[j])
                    
                # make final predictions
                x = x.view(x.size(0), -1)
                pred = x
                for j in range(7):
                    pred = self.classifier[j](pred)
                    if i == self.unroll_count - 1:
                        if k < 1000:
                            dprime_parameters[j+31][k][0] = torch.mean(pred).data.cpu().numpy()
                            dprime_parameters[j+31][k][1] = torch.std(pred).data.cpu().numpy()
                        if k == 1000:
                            np.save('/data2/simingy/data/dprime/recurrent_l3_u4_noise50/recurrent_l3_u4_noise50_layer{}.npy'.format(str(j+31)), dprime_parameters[j+31])


                #pred = self.classifier(x)
                intermediate_pred.append(pred)
                #print("Done4!")
        k = k + 1
        # collect predictions
        # batch_size * unroll_count * num_class
        intermediate_pred = torch.stack(intermediate_pred, dim=1)
        return intermediate_pred


if __name__ == '__main__':
    # high_layer, low_layer, high_layer_dim, low_layer_dim, scale_factor
    connections = (
        (6, 3, 128, 64, 2),
        (13, 8, 256, 128, 2),
        (20, 15, 512, 256, 2),
        (27, 22, 512, 512, 2)
    )

    # works for imagenet images
    model = MultipleRecurrentModel(network_cfg, connections, 5, 1000)
    print(model)
    model_input = A.Variable(torch.zeros(5, 3, 224, 224))
    model_output = model(model_input)
    print(model_output.size())
