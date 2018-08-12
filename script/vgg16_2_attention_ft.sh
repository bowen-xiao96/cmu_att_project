# variant vgg16 with vgg weight initialization
# no bn layer
# contains attention layer
# do gradient clipping

#python train.py --gpu 1 --expId vgg16_2_attention_ft --init_lr 1e-5 --network_config vgg16_2_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 0 --init_weight vgg --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_noattention/model-90.pth --fix_load_weight 0

python train.py --gpu 1 --expId vgg16_2_attention_ft --init_lr 1e-5 --network_config vgg16_2_attention.cfg --lr_freq 30 --adjust_lr 0 --optim 0 --init_weight vgg --data_path /data2/simingy/data/ --load_file /data2/simingy/model/vgg16_2_attention_ft/best.pkl --fix_load_weight 0
