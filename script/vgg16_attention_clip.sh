# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 1 --init_lr 0.001 --network_config vgg16_attention.cfg --lr_freq 200 --grad_clip 5 --save_dir /data2/simingy/model/vgg16_attention_clip --data_path /data2/simingy/data/

