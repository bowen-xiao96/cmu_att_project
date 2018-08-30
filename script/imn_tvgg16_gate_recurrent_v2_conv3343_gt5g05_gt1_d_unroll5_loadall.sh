# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --batch_size 64 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g05_gt1_d_unroll5_loadall --init_lr 1e-5 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g05_d_unroll5.cfg --lr_freq 300 --optim 4 --init_weight vgg --gate 1 --task gate_recurrent_v2_1 --dataset imagenet --load_vgg16 2 --weight_decay 5e-4 --data_path /data2/simingy/data/Imagenet



