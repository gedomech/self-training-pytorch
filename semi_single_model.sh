#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_0 --run_pretrain=True \
--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=100 --labeled_percentate=0.04 --idx_model=0 --performance_thres=0.5

OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_oracle/update_both/use_ce --run_semi=True \
--model_path runs/demo_self_training/pretrain_0.04/enet_0/best.pth --semi_train__lr=0.0001



OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/semi_update_both/enet_2/use_oracle --run_semi=True \
--model_path runs/demo_self_training/pretrain_0.04/enet_2/best.pth --semi_train__lr=0.0001 --semi_train__gamma=0.95 --batch_size=4 --idx_model=2 \
--semi_train__loss_name=oracle --semi_train__update_labeled=True --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04


OMP_NUM_THREADS=4  CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/semi_update_both/enet_2/use_ce --run_semi=True  \
--model_path runs/demo_self_training/pretrain_0.04/enet_2/best.pth --semi_train__lr=0.0001 --semi_train__gamma=0.95 --batch_size=4 --idx_model=2 \
--semi_train__loss_name=crossentropy --semi_train__update_labeled=True --semi_train__update_unlabeled=True --semi_train__max_epoch=100 --labeled_percentate=0.04




#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_1 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=1 --performance_thres=0.5
#
#
#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_2 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=2 --performance_thres=0.5
