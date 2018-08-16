# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 3 --init_lr 1e-4 --expId print_fe_test --network_config vgg16_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 1 --init_weight vgg --print_fe 1 --load_file /data2/simingy/model/vgg16_attention_ft_1/best.pkl

