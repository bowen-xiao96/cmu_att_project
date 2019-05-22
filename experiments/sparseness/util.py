import sys
import numpy as np
import torch

sys.path.insert(0, '/home/simingy/cmu_att_project/')
from model.get_model import get_model

## some utility functions for getting sparseness values

def load_up_model(name, weight_file):
    model = get_model(name)
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    del state_dict 
    return model

def population_sparseness(activations):
    """
    Compute the average population sparseness (sparseness across all units for each image, 
    averaged over all images) for a list of activation tensors

    Measure of sparseness used is the coefficient of variation -- higher is more sparse
    """
    b, c, w, h = activations[0].shape # batch size, channels, width, and height

    batch_sparses = []
    for batch in activations:
        mu = batch.mean(1).mean(1).mean(1) # across all but images
        sigma = batch.std(1).mean(1).mean(1) # stddev across channels, averaged over locations
        c_of_v = sigma / mu

        batch_sparses.append(c_of_v.mean())

    return float(np.mean(batch_sparses))

def lifetime_sparseness(activations):
    """
    Compute the average lifetime sparseness (sparseness across all images for each channel, 
    averaged over all channels) for a list of activation tensors

    Measure of sparseness used is the coefficient of variation -- higher is more sparse
    """
    b, c, w, h = activations[0].shape # batch size, channels, width, and height

    # this isn't right yet -- actually need to concatenate all batches together to get proper stddev
    # will fix soon
    batch_sparses = []
    for batch in activations:
        mu = batch.mean(0).mean(1).mean(1) # across all but channels
        sigma = batch.std(0).mean(1).mean(1)
        c_of_v = sigma / mu

        batch_sparses.append(c_of_v.mean())

    return float(np.mean(batch_sparses))
