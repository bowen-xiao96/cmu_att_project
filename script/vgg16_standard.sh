# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --init_lr 0.05 --network_config vgg16_nobn.cfg --save_dir /home/simingy/model/vgg16_nobn_standard --data_path /home/simingy/data/