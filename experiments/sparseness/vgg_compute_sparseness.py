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

model = util.load_up_model(model_name, model_file)


# load in all of imagenet -- takes a while
imagenet_dir = '/data2/simingy/data/Imagenet'
train_loader, test_loader = get_dataloader(imagenet_dir, 32, 4) 
# (maybe too small batchsize + worker count?)


# set things up to save the activations at the desired layers (each maxpool one for now)
layers = [4, 9, 16, 23, 30]
activations = dict.fromkeys(layers, [])
for layer in layers:
    # each forward pass will store the activations of each layer in the dictionary
    model.features[layer].register_forward_hook(lambda m, i, o: activations[layer].append(o))


# now actually run a thousand or so images through the network
# not sure if the data is shuffled -- might not be a random sample of images which would be bad
for x, y in itertools.islice(test_loader, 32):
    x = A.Variable(x.cuda(), volatile=True) # not doing backprop so volatile

    _ = model(x) # only care about the intermediate activations

# compute coefficients of variation (measure of sparseness)
pop_sparses = {}
life_sparses = {}
for layer in layers:
    pop_sparses[layer] = util.population_sparseness(activations[layer])
    life_sparses[layer] = util.lifetime_sparseness(activations[layer])

# save sparsenesses (temporarily, just here in the directory)
pop_file = open('vgg_pop_sparsenesses.pkl', 'wb')
pkl.dump(pop_sparses, pop_file)
pop_file.close()

life_file = open('vgg_life_sparsenesses.pkl', 'wb')
pkl.dump(life_sparses, life_file)
life_file.close()

# one thing I'm not quite sure of yet is whether the maxpool layer is the right/interesting one
# (maybe it should be the highest convolution layer in each stage)
