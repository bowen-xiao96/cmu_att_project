# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --init_lr 0.05 --expId vgg16_standard_cifar --network_config vgg16_nobn.cfg --data_path /home/simingy/data/
