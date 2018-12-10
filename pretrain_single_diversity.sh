#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 python train_onemodel.py --save_dir=demo_diversity/pretrain_1.0/enet --run_pretrain=True \
--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=1.0 --arch=enet


#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 python train_onemodel.py --save_dir=demo_diversity/pretrain_1.0/unet --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=1.0 --arch=unet


#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 python train_onemodel.py --save_dir=demo_diversity/pretrain_1.0/segnet --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=200 --labeled_percentate=1.0 --arch=segnet

