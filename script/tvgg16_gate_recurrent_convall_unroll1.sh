# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 3 --expId tvgg16_gate_recurrent_convall_unroll1 --init_lr 1e-4 --network_config tvgg16_gate_recurrent_convall_unroll1.cfg --lr_freq 30 --optim 1 --init_weight vgg --task gate_recurrent_noloss --data_path /data2/simingy/data/

#python train.py --gpu 3 --expId tvgg16_gate_recurrent_conv3343_unroll1 --init_lr 5e-5 --network_config tvgg16_gate_recurrent_conv3343_unroll1.cfg --lr_freq 30 --optim 1 --init_weight vgg --task gate_recurrent --data_path /data2/simingy/data/ --load_file /data2/simingy/model/tvgg16_gate_recurrent_conv3343_unroll1/best.pkl 
