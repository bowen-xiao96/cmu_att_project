# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

#python train.py --gpu 4,5,6,7 --expId vgg16_attention_recurrent_middle_unroll5 --init_lr 1e-4 --network_config vgg16_att_recurrent_middle_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /mnt/fs1/chengxuz/siming/Dataset/ --save_dir /mnt/fs1/chengxuz/siming/model/

python train.py --gpu 2,3 --expId vgg16_attention_recurrent_middle_unroll5 --init_lr 1e-4 --network_config vgg16_att_recurrent_middle_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/ 