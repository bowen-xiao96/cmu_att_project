# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

'''1e-5->1e-6->1e-7'''
python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt33g1_gt1_dd_unroll1_loadall --init_lr 1e-5 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt33g1_dd_unroll1.cfg --lr_freq 2 --optim 4 --init_weight vgg --gate 1 --task gate_recurrent_v2_1 --dataset imagenet --load_vgg16 2 --data_path /data2/simingy/data/Imagenet



