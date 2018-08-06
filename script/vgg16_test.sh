# standard vgg16 without any initialization 
# containing bn layer
# no dropout layer

python train.py --gpu 3 --init_lr 1e-1 --network_config vgg16.cfg
