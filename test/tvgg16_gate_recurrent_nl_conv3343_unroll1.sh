# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --expId test_tvgg16_gate_recurrent_nl_conv3343_unroll1 --init_lr 1e-10 --network_config tvgg16_gate_recurrent_nl_conv3343_unroll1.cfg --data_path /home/simingy/data/ --test_model 1 --add_noise 1 --task gate_recurrent_noloss --load_file /data2/simingy/model/tvgg16_gate_recurrent_nl_conv3343_unroll1/best.pkl