# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 3 --expId vgg16_attention --init_lr 0.05  --network_config vgg16_attention.cfg --lr_freq 30 --data_path /data2/simingy/data/