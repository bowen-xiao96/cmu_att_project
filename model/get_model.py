import os, sys
import torch
import torch.nn as nn
from torchvision.models import vgg16

import multiple_recurrent_l as l
import multiple_recurrent_newloss as nl
import multiple_recurrent_newgate as ng
import multiple_recurrent_nosplit_1 as ns_1
import gate as g

all_connections = (
    (13, 8, 256, 128, 2),
    (20, 15, 512, 256, 2),
    (27, 22, 512, 512, 2),
)

nosplit_connections = (
    (15, 8, 256, 128, 2),
    (20, 15, 512, 256, 2),
    # (27, 22, 512, 512, 2)
)

def get_model(name):
    # vgg, vgg_caffe (baseline models)
    # multiple_recurrent_l3 (connection 1+2)
    # multiple_recurrent_l4 (connection 2+3)
    # multiple_recurrent_newloss (connection 1+2, final loss)
    # loc1, loc2, loc3 (single connection)
    # gate1, gate2, gate3, gate4 (l3 model with different gates)
    # connect3 (three connections)

    unroll_count = 5
    num_class = 1000
    if name == 'vgg' or name == 'vgg_caffe':
        model = vgg16(num_classes=num_class, init_weights=False)

    elif name == 'multiple_recurrent_l3':
        model = l.MultipleRecurrentModel(l.network_cfg, all_connections[:-1], unroll_count, num_class)

    elif name == 'multiple_recurrent_l4':
        model = l.MultipleRecurrentModel(l.network_cfg, all_connections[1:], unroll_count, num_class)

    elif name == 'multiple_recurrent_newloss':
        model = nl.MultipleRecurrentModel(nl.network_cfg, all_connections[:-1], unroll_count, num_class, 'final')

    elif name.startswith('loc'):
        connection_idx = int(name[-1]) - 1
        model = l.MultipleRecurrentModel(l.network_cfg, (all_connections[connection_idx], ), unroll_count, num_class)

    elif name.startswith('gate'):
        gate_name = 'GatingModule%d' % int(name[-1])
        model = ng.MultipleRecurrentModel(ng.network_cfg, all_connections[:-1], unroll_count, num_class,
                                          gating_module=getattr(g, gate_name))
    
    elif name == 'connect3':
        model = l.MultipleRecurrentModel(l.network_cfg, all_connections, unroll_count, num_class)
    elif name == "multiple_recurrent_nosplit_1":
        model = ns_1.MultipleRecurrentModel(ns_1.network_cfg, nosplit_connections, unroll_count, num_class, gating_module=getattr(g, 'GatingModule8'))
    else:
        raise NotImplementedError

    return model
