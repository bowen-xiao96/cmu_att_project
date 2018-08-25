# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 3 --expId tvgg16_gate_recurrent_nl_conv3343_gt5_gt1_lrelu_unroll5 --init_lr 2e-4 --network_config tvgg16_gate_recurrent_nl_conv3343_gt5_lrelu_unroll5.cfg --lr_freq 150 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_noloss --data_path /data2/simingy/data/
