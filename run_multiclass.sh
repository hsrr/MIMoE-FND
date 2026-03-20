#!/bin/bash

python train_multiclass.py \
    -train_file /data1/hsiri/AMG/datasets/trianDBthinkGeminiCOT3_260107.jsonl \
    -val_file /data1/hsiri/AMG/datasets/val.json \
    -train_image_root /data1/hsiri/AMG/datasets/AMG_MEDIA/train_imagesN \
    -val_image_root /data1/hsiri/AMG/datasets/AMG_MEDIA/val_imagesN \
    -dataset_name Twitter \
    -device cuda:0 \
    -batch_size 16 \
    -epochs 100 \
    -finetune 0 \
    -int_lr 1e-4 \
    -int_beta 0.7 \
    -agr_threshold 0.3 \
    -sem_threshold 0.3 \
    -max_words 512 \
    -output_dir ./checkpoints/multiclass
