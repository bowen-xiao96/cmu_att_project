# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer

python train.py --gpu 2 --expId tvgg16_gate_recurrent_conv3343_a9g9_unroll5 --init_lr 1e-4 --network_config tvgg16_gate_recurrent_conv3343_a9g9_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --task gate_recurrent --data_path /data2/simingy/data/

#python train.py --gpu 2 --expId tvgg16_gate_recurrent_conv3343_a9g9_unroll5 --init_lr 1e-4 --network_config tvgg16_gate_recurrent_conv3343_a9g9_unroll5.cfg --lr_freq 30 --optim 1 --init_weight vgg --task gate_recurrent --data_path /data2/simingy/data/ --load_file /data2/simingy/model/tvgg16_gate_recurrent_conv3343_a9g9_unroll5/best.pkl 


