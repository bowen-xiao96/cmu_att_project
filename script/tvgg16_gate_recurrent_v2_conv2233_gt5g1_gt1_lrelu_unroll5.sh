# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 1 --expId tvgg16_gate_recurrent_v2_conv2233_gt5g1_gt1_lrelu_unroll5 --init_lr 2e-4 --network_config tvgg16_gate_recurrent_v2_conv2233_gt5g1_lrelu_unroll5.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --data_path /data2/simingy/data/