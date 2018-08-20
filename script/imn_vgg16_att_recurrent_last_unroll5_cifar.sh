# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention recurrent layer


python train.py --gpu 0,1,2,3 --expId imn_vgg16_attention_recurrent_last_unroll5_cifar_1 --batch_size 128 --init_lr 1e-4 --network_config imn_vgg16_att_recurrent_last_unroll5.cfg --task imagenet --lr_freq 20 --optim 1 --display_freq 200 --save_every 5 --init_weight vgg --data_path /data2/simingy/data/Imagenet --load_file /data2/simingy/model/vgg16_attention_recurrent_last_unroll5/best.pkl --load_part_params 1

