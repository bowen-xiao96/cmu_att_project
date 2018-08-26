# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --expId test_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_lrelu_unroll1 --init_lr 1e-10 --network_config tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_lrelu_unroll1.cfg --data_path /home/simingy/data/ --gate 1 --test_model 1 --add_noise 0 --task gate_recurrent_v2 --load_file /data2/simingy/model/tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_lrelu_unroll1/best.pkl

