# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0 --expId vvgg16_attention_recurrent_middle_unroll5 --init_lr 1e-4 --network_config vvgg16_att_recurrent_middle_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/

#python train.py --gpu 0,1 --expId vgg16_attention_recurrent_last_unroll5 --init_lr 1e-4 --network_config vgg16_att_recurrent_last_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/ 
