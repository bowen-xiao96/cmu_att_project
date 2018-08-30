# standard vgg16 without any initialization 
# no bn layer

for ((i=10; i <=50; i+=10))
do
    python train.py --gpu 0,1,2,3 --batch_size 128 --expId test_imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1 --init_lr 1e-10 --optim 1 --network_config imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_unroll1.cfg --data_path /data2/simingy/data/Imagenet --dataset noise_imagenet --gate 1 --test_model 1 --noise_level $i --task gate_recurrent_v2 --load_file /data2/simingy/model/imn_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_unroll1_loadall_2/best.pkl
done

