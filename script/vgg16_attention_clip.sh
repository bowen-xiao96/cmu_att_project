# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

python train.py --gpu 1 --init_lr 0.00025 --network_config vgg16_attention.cfg --lr_freq 30 --adjust_lr 0 --grad_clip 5 --init_weight None --save_dir /data2/simingy/model/vgg16_attention_clip --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_attention_clip/model-180.pth

