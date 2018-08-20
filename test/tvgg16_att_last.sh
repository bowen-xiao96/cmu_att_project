#python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/tvgg16_attention_last/best.pkl --network_config tvgg16_attention_last.cfg --expId test_tvgg16_attention_last_cifar10 --dataset cifar10 --task attention

python test_on_imagenet.py --gpu 0 --load_file /data2/simingy/model/tvgg16_attention_last/best.pkl --network_config tvgg16_attention_last.cfg --expId test_tvgg16_attention_last_imagenet --dataset imagenet --task attention

