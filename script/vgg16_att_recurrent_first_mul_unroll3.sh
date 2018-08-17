# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --expId vgg16_attention_recurrent_first_mul_unroll3 --init_lr 1e-4 --network_config vgg16_att_recurrent_roll3.cfg --lr_freq 30 --optim 3 --init_weight vgg --att_r_type 1 --data_path /mnt/fs1/chengxuz/siming/Dataset/ --save_dir /mnt/fs1/chengxuz/siming/model/ 
#--load_file /mnt/fs1/chengxuz/siming/model/vgg16_attention_recurrent_first/best.pkl

