#!bin/sh
for file in /data2/simingy/data/Imagenet/val/*
do
    if test -f $file
    then
        echo $file 是文件
    fi
    if test -d $file
    then
        python train_vgg16.py -1 /data2/simingy/test/ $file
    fi
done
