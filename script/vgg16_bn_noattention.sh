# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 2 --init_lr 0.01 --network_config vgg16_bn_noattention.cfg --lr_freq 30 --init_weight vgg --save_dir /data2/simingy/model/vgg16_bn_attention --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_bn_attention/model-30.pth 
