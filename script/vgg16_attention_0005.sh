# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# init learning rate 0.001

python train.py --gpu 1 --init_lr 0.001 --network_config vgg16_attention.cfg --lr_freq 200 --save_dir /data2/simingy/model/vgg16_attention_001 --data_path /data2/simingy/data/
