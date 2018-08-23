# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --init_lr 0.05 --expId vgg16_standard --network_config vgg16_nobn_imn.cfg --data_path /home/simingy/data/
