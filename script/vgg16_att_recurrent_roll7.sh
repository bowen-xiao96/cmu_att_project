# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer
# roll 7 times

python train.py --gpu 0,1,2,3 --expId vgg16_attention_recurrent_roll_7 --init_lr 1e-4 --network_config vgg16_att_recurrent_roll7.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/
