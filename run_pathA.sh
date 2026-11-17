#!/bin/bash
# Path A Training Script - Transformer Block Upgrade
PYTHONPATH=. python3 scripts/FMPose3D_main.py \
  --train --test 1 --nepoch 81 \
  --batch_size 256 --lr 0.001 \
  --lr_decay 0.98 --lr_decay_large 0.8 --large_decay_epoch 5 \
  --gpu 0 --model fmpose3d --dataset h36m \
  --root_path ../FMPose3D_Improved/dataset/ \
  --test_augmentation 1 --test_augmentation_flip_hypothesis 1 \
  --hypothesis_num 8 --topk 8 --exp_temp 0.002 \
  --sample_steps 3 --sh_file run_pathA.sh
