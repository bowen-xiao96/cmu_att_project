for ((i=1; i <= 16; i*=2))
do
    python ./experiments/train_multiple_recurrent_l.py 0,1,2,3 5 /data2/simingy/model/ /data2/bowenx/recurrent_gating_model.pkl $i
done
