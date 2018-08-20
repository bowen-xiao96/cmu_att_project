# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 2 --expId vgg16_attention_middle --init_lr 0.05  --network_config vgg16_attention_middle.cfg --lr_freq 30 --init_weight vgg --data_path /data2/simingy/data/
