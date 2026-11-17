#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH=.
python3 scripts/FMPose3D_main.py \
  --train \
  --test 1 \
  --nepoch 81 \
  --batch_size 256 \
  --lr 0.001 \
  --gpu 0 \
  --model fmpose3d \
  --layers 5 \
  --channel 512 \
  --dataset h36m \
  --root_path ../FMPose3D_Improved/dataset/ \
  --workers 8 \
  --frames 1 \
  --pad 0 \
  --hypothesis_num 8 \
  --topk 8 \
  --exp_temp 0.002 \
  --sample_steps 5 \
  --sh_file run_train_pathA.sh
