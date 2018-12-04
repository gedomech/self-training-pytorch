#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/pretrain_0.004/enet_0 --run_pretrain=True \
--full_train__gamma=0.9 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=0

OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/update_both/use_ce\
 --run_semi=True  --model_path runs/demo_self_training/pretrain_0.04/best.pth --semi_train__lr=0.001 \
--semi_train__gamma=0.95 --batch_size=4 --semi_train__loss_name=crossentropy --semi_train__update_labeled=True\
 --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04

