# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 0 --init_lr 0.05 --network_config vgg16_noattention.cfg
