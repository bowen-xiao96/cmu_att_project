# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 0 --init_lr 0.01 --network_config vgg16_noattention.cfg --lr_freq 200 --save_dir /home/simingy/model/vgg16_noattention --data_path /home/simingy/data/
