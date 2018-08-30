# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

'''1e-5->1e-6->1e-7'''
#python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall --init_lr 1e-5 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --dataset imagenet --load_vgg16 2 --data_path /data2/simingy/data/Imagenet

#python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall_1 --init_lr 1e-6 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --dataset imagenet --data_path /data2/simingy/data/Imagenet --load_file /data2/simingy/model/imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall/best.pkl

#python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall_2 --init_lr 1e-7 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --lr_freq 300 --optim 1 --init_weight vgg --gate 1 --task gate_recurrent_v2 --dataset imagenet --data_path /data2/simingy/data/Imagenet --load_file /data2/simingy/model/imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall_1/best.pkl

'''2e-5->2e-6->2e-7'''
python train.py --gpu 0,1,2,3 --batch_size 128 --expId imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall_dl --init_lr 2e-5 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --optim 4 --init_weight vgg --gate 1 --task gate_recurrent_v2_1 --dataset imagenet --load_vgg16 2 --data_path /data2/simingy/data/Imagenet

