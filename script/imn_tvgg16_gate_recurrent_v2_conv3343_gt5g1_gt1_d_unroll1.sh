# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1 --init_lr 1e-5 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --dataset imagenet --load_vgg16 1 --data_path /data2/simingy/data/Imagenet
