#python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/tvgg16_attention_last/best.pkl --network_config tvgg16_attention_last.cfg --expId test_tvgg16_attention_last_cifar10 --dataset cifar10 --task attention

#python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/vgg16_attention_last/best.pkl --network_config vgg16_attention_last.cfg --expId test_vgg16_attention_last_cifar10 --dataset cifar10 --task attention

python train.py --gpu 2 --expId test_vgg16_attention_last --init_lr 1e-10 --network_config vgg16_attention_last.cfg --data_path /home/simingy/data/ --test_model 1 --add_noise 1 --task attention --load_file /data2/simingy/model/vgg16_attention_last/best.pkl


