import sys
import numpy as np
from collections import OrderedDict
from get_noise import get_dataloader

sys.path.insert(0, '/home/simingy/cmu_att_project/')

from model.get_model import get_model
from model.multiple_recurrent_l import *
from utils.metric import accuracy

if __name__ == '__main__':
    # fix the seed of the RNG
    np.random.seed(0)

    assert len(sys.argv) > 4
    GPU_ID = int(sys.argv[1])
    model_name = sys.argv[2]
    weight_file = sys.argv[3]
    noise_sigma = float(sys.argv[4])

    if GPU_ID == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

    # prapare model
    model = get_model(model_name)
    new_loss = model_name == 'multiple_recurrent_newloss'
    if model_name != 'vgg' and model_name != 'vgg_caffe':
        # recurrent model, need multiple GPUs
        model = nn.DataParallel(model)

    # load weights
    state_dict = torch.load(weight_file)
    if isinstance(state_dict, tuple):
        state_dict = state_dict[-1]

    if model_name == 'vgg_caffe':
        m = {'classifier.1.weight': 'classifier.0.weight', 'classifier.1.bias': 'classifier.0.bias',
             'classifier.4.weight': 'classifier.3.weight', 'classifier.4.bias': 'classifier.3.bias'}
        state_dict = OrderedDict([(m[k] if k in m else k, v) for k, v in state_dict.items()])

    model.load_state_dict(state_dict)
    model.eval()
    model = model.cuda()
    del state_dict
    print(model)
    # load data and test
    imagenet_dir = '/data2/simingy/data/Imagenet'
    mode = 'caffe' if model_name == 'vgg_caffe' else 'pytorch'
    test_loader = get_dataloader(imagenet_dir, mode, 256, 8, noise_sigma)

    pred = list()
    gt = list()

    for x, y in test_loader:
        x = A.Variable(x.cuda(), volatile=True)
        y = A.Variable(y.cuda(), volatile=True)

        if new_loss:
            output_ = model(x, y)
        else:
            output_ = model(x)

        if isinstance(output_, tuple):
            output, _ = output_
        else:
            output = output_

        pred.append(output.data.cpu())
        gt.append(y.data.cpu())

    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    prec1, prec5 = accuracy(pred, gt, topk=(1, 5))
    print(prec1[0], prec5[0])
