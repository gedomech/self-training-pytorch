#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_0 --run_pretrain=True \
--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=0


#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_1 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=1
#
#
#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train_onemodel.py --save_dir demo_self_training/pretrain_0.04/enet_2 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=0.04 --idx_model=2
