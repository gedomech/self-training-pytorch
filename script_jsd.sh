#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python jsd_semi_experiment.py --model_path=checkpoints --voting_strategy=hard --save_dir=jsd/hard \
--gamma=0.8 --lr=0.0001 --max_epoch=3 --num_workers=2 --batch_size=2 --loss_name=jsd &
CUDA_VISIBLE_DEVICES=1 python jsd_semi_experiment.py  --model_path=checkpoints --voting_strategy=soft --save_dir=jsd/soft \
--gamma=0.8 --lr=0.0001 --max_epoch=3 --num_workers=2 --batch_size=2 --loss_name=jsd