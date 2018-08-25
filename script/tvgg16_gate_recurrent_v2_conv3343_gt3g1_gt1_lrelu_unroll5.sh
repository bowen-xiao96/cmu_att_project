# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 2 --expId tvgg16_gate_recurrent_v2_conv3343_gt3g1_gt1_lrelu_unroll5 --init_lr 2e-4 --network_config tvgg16_gate_recurrent_v2_conv3343_gt3g1_lrelu_unroll5.cfg --lr_freq 150 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --data_path /data2/simingy/data/