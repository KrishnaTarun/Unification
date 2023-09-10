
# !/bin/




# Vic-Reg gating

listVar="0.1 0.2 0.3 0.4 0.5"
for f in 0 1
do

    for i in $listVar 
    # for i in {1..20..1} 
    do 
        echo "$i"

        CUBLAS_WORKSPACE_CONFIG=:16:8 python ../../../main_knn.py \
        --dataset stl10 \
        --data_dir /YOUR_PATH/Unification/bash_files/pretrain/datasets \
        --pretrained_checkpoint_dir /YOUR_PATH/Unification/bash_files/pretrain/stl10/trained_models/separate_bn/vicreg_gating/$i \
        --train_dir stl10/train \
        --val_dir stl10/val \
        --feature_type backbone \
        --batch_size 256 \
        --token 1 \
        --flag $f \
        --distance_function euclidean \
        --k 1

    #     # sleep 2m

    done
done
