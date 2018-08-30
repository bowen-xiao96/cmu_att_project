# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 0,1,2,3 --batch_size 128 --init_lr 0.05 --expId vgg16_standard_imn --network_config vgg16_nobn_imn.cfg --data_path /data2/simingy/data/Imagenet --dataset imagenet --load_vgg16 2 --test_model 1
