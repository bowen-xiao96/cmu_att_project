# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 3 --expId tvgg16_gate_recurrent_v2_conv3343_gt7g1_gt1_lrelu_unroll5 --init_lr 1e-4 --network_config tvgg16_gate_recurrent_v2_conv3343_gt7g1_lrelu_unroll5.cfg --lr_freq 150 --lr_decay 0.1 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --data_path /data2/simingy/data/ --load_file /data2/simingy/model/tvgg16_gate_recurrent_v2_conv3343_gt7g1_gt1_lrelu_unroll5/180.pkl 
