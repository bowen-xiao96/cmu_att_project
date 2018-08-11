# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 1 --expId vgg16_attention_ft_new --init_lr 5e-5 --network_config vgg16_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 1 --init_weight vgg --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_attention_ft_new/best.pkl --fix_load_weight 0
