# standard vgg16 without any initialization 
# no bn layer

python train.py --gpu 0,1,2,3 --batch_size 128 --expId test_imn_vgg16_standard --init_lr 1e-10 --optim 1 --network_config vgg16_nobn_imn.cfg --data_path /data2/simingy/data/Imagenet --dataset imagenet --test_model 1 --add_noise 1 --load_vgg16 2 
