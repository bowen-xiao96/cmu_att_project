#python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/vgg16_attention_recurrent_last_unroll5/best.pkl --network_config vgg16_att_recurrent_last_unroll5.cfg --expId test_tvgg16_att_recurrent_last_unroll5 --att_unroll_count 5

for i in $(seq 0 5)
do
    python test_on_imagenet.py --gpu 0 --expId test_tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_lrelu_unroll1 --network_config tvgg16_gate_recurrent_v2_conv3343_gt5g1_d_lrelu_unroll1.cfg --dataset cifar10 --task recurrent_gate_v2 --gate 1 --load_file /data2/simingy/model/tvgg16_gate_recurrent_v2_conv3343_gt5g1_gt1_d_lrelu_unroll1_again/$i.pkl
done

