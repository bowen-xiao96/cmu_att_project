# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 3 --expId tvgg16_gate_recurrent_v2_conv2233_gt5g1_gt1_d_lrelu_unroll1 --init_lr 2e-4 --network_config tvgg16_gate_recurrent_v2_conv2233_gt5g1_d_lrelu_unroll1.cfg --lr_freq 200 --lr_decay 0.1 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --data_path /data2/simingy/data/
