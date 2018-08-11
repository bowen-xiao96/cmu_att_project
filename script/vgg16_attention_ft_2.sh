# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 0 --init_lr 1e-4 --network_config vgg16_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 1 --init_weight vgg --save_dir /data2/simingy/model/vgg16_attention_ft_nofix --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_noattention/model-120.pth --fix_load_weight 0
