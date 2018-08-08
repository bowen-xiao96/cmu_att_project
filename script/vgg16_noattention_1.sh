# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 3 --init_lr 0.05 --network_config vgg16_noattention.cfg --lr_freq 30 --adjust_lr 1 --save_dir /data2/simingy/model/vgg16_noattention_1 --data_path /data2/simingy/data/
