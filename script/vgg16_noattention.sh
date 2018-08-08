# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 0 --init_lr 0.00125 --network_config vgg16_noattention.cfg --lr_freq 60 --adjust_lr 0 --save_dir /data2/simingy/model/vgg16_noattention --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_noattention/model-60.pth
