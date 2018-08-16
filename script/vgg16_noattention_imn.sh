# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 0,1,2,3 --expId vgg16_noattention_imn --batch_size 64 --init_lr 0.01 --network_config vgg16_noattention_imn.cfg --task imagenet --data_path /data2/simingy/data/Imagenet
