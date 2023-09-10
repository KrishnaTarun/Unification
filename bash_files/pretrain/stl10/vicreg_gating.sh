#!/bin/bash

# vicreg_gating_dual 


listVar="0.1 0.2 0.3 0.4 0.5"
for i in $listVar
do 
    echo "$i"

    python3 ../../../main_pretrain_prun.py \
        --dataset stl10 \
        --backbone resnet18 \
        --data_dir ../datasets \
        --train_dir stl10/train \
        --val_dir stl10/val \
        --max_epochs 500 \
        --gpus 0,1 \
        --accelerator ddp \
        --sync_batchnorm \
        --precision 16 \
        --optimizer sgd \
        --lars \
        --grad_clip_lars \
        --eta_lars 0.02 \
        --exclude_bias_n_norm \
        --scheduler warmup_cosine \
        --scheduler warmup_cosine \
        --lr 0.3 \
        --weight_decay 1e-4 \
        --batch_size 256 \
        --num_workers 4 \
        --crop_size 96 \
        --min_scale 0.2 \
        --brightness 0.4 \
        --contrast 0.4 \
        --saturation 0.2 \
        --hue 0.1 \
        --solarization_prob 0.1 \
        --gaussian_prob 0.0 0.0 \
        --num_crops_per_aug 1 1 \
        --project stl10-gating \
        --entity tkrishna \
        --wandb \
        --save_checkpoint \
        --den-target $i \
        --lbda 5 \
        --gamma 1 \
        --alpha 2e-2 \
        --method vicreg_gating \
        --name vicreg_gating_dual \
        --proj_hidden_dim 2048 \
        --proj_output_dim 2048 \
        --sim_loss_weight 25.0 \
        --var_loss_weight 25.0 \
        --cov_loss_weight 1.0 \
        --width 64 \
        --separate_bn \
        
done
# --knn_eval \