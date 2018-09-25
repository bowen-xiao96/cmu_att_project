import os, sys
import subprocess
import pickle

noise_sigmas = (0, 10, 20, 30, 40, 50)
models = ('vgg',
          'multiple_recurrent_l3',
          'multiple_recurrent_newloss',
          'loc1',
          'loc2',
          'loc3',
          'gate1',
          'gate2',
          'gate3',
          'connect3')

output_dict = dict()
for model in models:
    if model == 'vgg':
        model_file = '/data2/bowenx/visual_concepts/model/vgg16-397923af.pth'
    elif model == 'connect3':
        model_file = '/data2/bowenx/connection_3_siming.pkl'
    else:
        model_file = '../%s/best.pkl' % model

    for sigma in noise_sigmas:
        key = (model, sigma)
        print(key)

        try:
            output = subprocess.check_output(['python', 'noise_test.py', str(-1), model, model_file, str(sigma)])

            output = output.decode('utf-8')
            top1, top5 = output.split()
            output_dict[key] = (float(top1), float(top5))
        except subprocess.CalledProcessError as ex:
            print('Error for model %s and sigma %d' % (model, sigma))

with open('output.pkl', 'wb') as f_out:
    pickle.dump(output_dict, f_out, pickle.HIGHEST_PROTOCOL)
