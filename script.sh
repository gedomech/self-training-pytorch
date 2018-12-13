#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python jsd_semi_experiment.py  --model_path=checkpoints --voting_strategy=hard --save_dir=hard &
CUDA_VISIBLE_DEVICES=1 python jsd_semi_experiment.py  --model_path=checkpoints --voting_strategy=soft --save_dir=soft