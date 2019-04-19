import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as A


def my_PCA(data, k=2):
    # preprocess the data
    X = data

    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)

    # svd
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])


