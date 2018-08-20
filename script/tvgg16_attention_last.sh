# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 2 --expId tvgg16_attention_last --init_lr 0.05 --network_config tvgg16_attention_last.cfg --init_weight vgg --lr_freq 30 --data_path /data2/simingy/data/
