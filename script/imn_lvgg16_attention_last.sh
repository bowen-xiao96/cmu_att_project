# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer

python train.py --gpu 0,1,2,3 --expId imn_lvgg16_attention_last --batch_size 128 --init_lr 0.001 --network_config imn_lvgg16_attention_last.cfg --task imagenet --optim 1 --init_weight vgg --lr_freq 20 --save_every 5 --display_freq 200 --data_path /data2/simingy/data/Imagenet 
