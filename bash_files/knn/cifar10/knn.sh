
# !/bin/


# Vic-Reg gating:

listVar="0.1 0.2 0.3 0.4 0.5"
for i in $listVar 
# for i in {1..20..1} 
do 
    echo "$i"

    python ../../../main_knn.py \
    --dataset cifar10 \
    --data_dir /YOUR_PATH/Unification/bash_files/pretrain/datasets/ \
    --pretrained_checkpoint_dir /YOUR_PATH/Unification/bash_files/pretrain/cifar10/trained_models/vicreg_gating/$i \
    --train_dir cifar10/train \
    --val_dir cifar10/val \
    --feature_type backbone \
    --batch_size 256 \
    --token 1 \
    --flag 0 \
    --distance_function euclidean \
    --k 1

    # sleep 2m

done
