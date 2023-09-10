
# !/bin/




# Vic-Reg gating
listVar="0.3 0.4 0.5"
for i in $listVar 
# for i in {1..20..1} 
do 
    echo "$i"

    CUBLAS_WORKSPACE_CONFIG=:16:8 python ../../../main_knn.py \
    --dataset imagenet100 \
    --data_dir /YOUR_PATH/Unification/bash_files/pretrain/datasets/ \
    --pretrained_checkpoint_dir /YOUR_PATH/Unification/bash_files/pretrain/imagenet100/trained_models/vicreg_gating_separate_bn/$i \
    --train_dir imagenet100/train \
    --val_dir imagenet100/val \
    --feature_type backbone \
    --batch_size 256 \
    --token 1 \
    --flag 0 \
    --distance_function euclidean \
    --k 1

#     # sleep 2m

done
