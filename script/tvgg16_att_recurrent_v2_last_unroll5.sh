# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --expId tvgg16_attention_recurrent_v2_last_unroll5 --init_lr 1e-4 --network_config tvgg16_att_recurrent_v2_last_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/


