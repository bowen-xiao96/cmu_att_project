# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 0,1,2,3 --batch_size 32 --expId imn_tvgg16_gate_recurrent_nl_conv3343_gt5_gt1_lrelu_unroll1 --init_lr 2e-4 --network_config imn_tvgg16_gate_recurrent_nl_conv3343_gt5_lrelu_unroll1.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_noloss --dataset imagenet --data_path /data2/simingy/data/Imagenet
