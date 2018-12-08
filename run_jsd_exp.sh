#!/usr/bin/env bash

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python jsd_experiment.py --save_dir demo_jsd/pretrain_0.04/enet_0 --run_pretrain=True \
--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=100 --labeled_percentate=0.04 --seed=1 &


#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python jsd_experiment.py --save_dir demo_jsd/pretrain_0.04/enet_0 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=100 --labeled_percentate=0.04 --seed=1


#OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python jsd_experiment.py --save_dir demo_jsd/pretrain_0.04/enet_0 --run_pretrain=True \
#--full_train__gamma=0.8 --full_train__lr=0.001 --full_train__max_epoch=100 --labeled_percentate=0.04 --seed=1


