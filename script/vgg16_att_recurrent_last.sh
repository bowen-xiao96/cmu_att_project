# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --expId vgg16_attention_recurrent_last_test --init_lr 1e-4 --network_config vgg16_att_recurrent_last.cfg --lr_freq 30 --adjust_lr 0 --optim 2 --init_weight vgg --data_path /mnt/fs1/chengxuz/siming/Dataset/ --save_dir /mnt/fs1/chengxuz/siming/model/
