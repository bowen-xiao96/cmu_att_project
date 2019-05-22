import os
import numpy as np


temp_list = [8, 9, 10, 11, 13, 15, 16, 18] 
temp_dic = {'8': 0, '9': 1, '10': 2, '11': 3, '13': 4, '15': 5, '16': 6, '18': 7, '20': 8, '22': 9, '23': 10, '25': 11, '27': 12, '29': 13}

for k in range(38):
    data_path = '/data2/simingy/data/dprime/recurrent_l3_u4_noise50/recurrent_l3_u4_noise50_layer{}.npy'.format(str(k))
    dprime_num = np.zeros((1000, 1000))

    datas = np.load(data_path)

    for i in range(1000):
        for j in range(1000):
            dprime_num[i][j] = np.true_divide((np.abs(datas[i][0] - datas[j][0])), np.sqrt(0.5 * (np.square(datas[i][1]) + np.square(datas[j][1]))))

    np.save('/data2/simingy/data/dprime/recurrent_l3_u4_noise50/dprime_l3_u4_recurrent_noise50_layer{}.npy'.format(str(k)), dprime_num)


