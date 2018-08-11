# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 1 --expId vgg16_2_attention_ft --init_lr 1e-4 --network_config vgg16_2_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 0 --init_weight vgg --data_path /data2/simingy/data/ --fix_load_weight 0
