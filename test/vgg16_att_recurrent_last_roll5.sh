#python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/vgg16_attention_recurrent_last_unroll5/best.pkl --network_config vgg16_att_recurrent_last_unroll5.cfg --expId test_tvgg16_att_recurrent_last_unroll5 --att_unroll_count 5

python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/vvgg16_attention_recurrent_last_unroll5/60.pkl --network_config vvgg16_att_recurrent_last_unroll5.cfg --expId test_vgg16_att_recurrent_last_unroll5_cifar10 --att_unroll_count 5 --dataset cifar10 --task recurrent_att

