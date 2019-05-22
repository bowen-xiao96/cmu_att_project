import sys
import util
import itertools
import torch.autograd as A
import pickle as pkl

sys.path.insert(0, '/home/simingy/cmu_att_project/')

from model.get_model import get_model
from dataset.imagenet.get_imagenet_dataset import get_dataloader

# this is mostly copied from files in the noise experiment

model_file = '/data2/bowenx/visual_concepts/model/vgg16-397923af.pth'
model_name = 'vgg' # for now
