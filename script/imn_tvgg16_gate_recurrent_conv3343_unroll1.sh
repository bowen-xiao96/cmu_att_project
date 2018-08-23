# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_nl_conv3343_unroll1 --init_lr 1e-4 --network_config imn_tvgg16_gate_recurrent_nl_conv3343_unroll1.cfg --lr_freq 20 --save_every 5 --optim 1 --init_weight vgg --task gate_recurrent_noloss --data_path /data2/simingy/data/Imagenet --dataset imagenet 

