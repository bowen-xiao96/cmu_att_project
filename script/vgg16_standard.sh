# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 2 --init_lr 0.05 --network_config vgg16_nobn.cfg --save_file /mnt/fs1/siming/model/vgg16_nobn_cifar10.pth
