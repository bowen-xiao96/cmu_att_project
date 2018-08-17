# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

#python train.py --gpu 0,1,2,3 --expId vgg16_attention_recurrent_last --init_lr 1e-4 --network_config vgg16_att_recurrent_last.cfg --lr_freq 30 --optim 1 --init_weight vgg --att_channel 512  --data_path /mnt/fs1/chengxuz/siming/Dataset/ --save_dir /mnt/fs1/chengxuz/siming/model/

python train.py --gpu 0,1,2,3 --expId vgg16_attention_recurrent_first_c --init_lr 1e-4 --network_config vgg16_att_recurrent.cfg --lr_freq 30 --optim 1 --init_weight vgg --att_channel 256 --data_path /data2/simingy/data/ 
