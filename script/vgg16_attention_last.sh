# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 3 --expId vgg16_attention_last --init_lr 0.05  --network_config vgg16_attention_last.cfg --lr_freq 30 --init_weight vgg --data_path /data2/simingy/data/
