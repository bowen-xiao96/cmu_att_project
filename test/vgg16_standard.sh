# standard vgg16 without any initialization 
# no bn layer

#python train.py --gpu 3 --expId test_vgg16_standard_cifar --init_lr 1e-10 --network_config vgg16_nobn.cfg --data_path /home/simingy/data/ --test_model 1 --add_noise 1 --load_file /data2/simingy/model/vgg16_standard_cifar/best.pkl

for i in $(seq 0 5)
do
     python test_on_imagenet.py --gpu 0 --expId test_vgg16_standard_cifar --init_lr 1e-10 --network_config vgg16_nobn.cfg --dataset cifar10 --task recurrent_gate_v2 --load_file /data2/simingy/model/vgg16_standard_cifar/$i.pkl
done
   
