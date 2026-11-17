#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from fmpose3d.common.h36m_dataset import Human36mDataset
from fmpose3d.common.load_data_hm36 import Fusion
from fmpose3d.common.utils import mpjpe, p_mpjpe
from fmpose3d.models import build_model
from fmpose3d.aggregation_methods import aggregation_RPEA_joint_level

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='h36m', type=str)
    parser.add_argument('--data_root', default='./dataset/', type=str)
    parser.add_argument('--model', default='fmpose3d_humans', type=str)
    parser.add_argument('--nepoch', default=80, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_works', default=4, type=int)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--layers', default=5, type=int)
    parser.add_argument('--pad', default=1, type=int)
    parser.add_argument('--train_views', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--test_views', type=int, nargs='+', default=[0,1,2,3])
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--use_pathA', default=True, type=bool)
    parser.add_argument('--d_hid', default=512, type=int)
    parser.add_argument('--token_dim', default=512, type=int)
    parser.add_argument('--n_joints', default=17, type=int)
    parser.add_argument('--exp_temp', default=0.002, type=float)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--num_hypotheses', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
