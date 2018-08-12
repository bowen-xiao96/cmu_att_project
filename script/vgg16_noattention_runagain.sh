# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 3 --expId vgg16_noattention_runagain --init_lr 0.05 --network_config vgg16_noattention.cfg --data_path /data2/simingy/data/ 
