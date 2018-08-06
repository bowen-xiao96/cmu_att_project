# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 2 --init_lr 0.001 --network_config vgg16_attention.cfg
