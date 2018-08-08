# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 3 --init_lr 0.0001 --network_config vgg16_attention.cfg --lr_freq 200 --save_dir /data2/simingy/model/vgg16_attention --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_attention/model-30.pth
