# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 3 --init_lr 0.0001 --network_config vgg16_attention.cfg --lr_freq 200
