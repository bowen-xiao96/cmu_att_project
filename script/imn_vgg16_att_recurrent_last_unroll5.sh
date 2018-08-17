# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0 --expId imn_vgg16_attention_recurrent_last_unroll5 --batch_size 4 --init_lr 1e-4 --network_config imn_vgg16_att_recurrent_last_unroll5.cfg --task imagenet --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/Imagenet 

#python train.py --gpu 0 --expId imn_vgg16_attention_recurrent_last_unroll5 --batch_size 4 --init_lr 1e-4 --network_config vgg16_noattention_imn.cfg --task imagenet --lr_freq 30 --optim 1 --init_weight vgg --data_path /data2/simingy/data/Imagenet 
