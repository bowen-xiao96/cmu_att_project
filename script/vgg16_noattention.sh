# variant vgg16 with vgg weight initialization
# no bn layer
# no attention layer

python train.py --gpu 0 --init_lr 0.05 --network_config vgg16_noattention.cfg --save_dir ./model/vgg16_noattention --data_path ./data/
