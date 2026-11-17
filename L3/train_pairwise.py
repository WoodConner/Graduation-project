#!/usr/bin/env python3
"""
L3 Training with Pairwise Distance Features (272-dim)
Invariant to joint ordering, scale, and orientation.
"""
import argparse, os, sys, logging, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from L3.dataset import H36M_ACTIONS, H36M_TRAIN_SUBJECTS, H36M_TEST_SUBJECTS

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def seq_to_pairwise_features(seq):
    """
    Convert (T, 17, 3) sequence to (T, 136) pairwise distance features.
    136 = C(17, 2) = 17*16/2 upper triangle distances.
    Invariant to joint ordering, translation, rotation.
    """
    T = seq.shape[0]
    feats = []
    for t in range(T):
        # Distance matrix: (17, 17)
        dists = np.linalg.norm(seq[t][:, None] - seq[t][None, :], axis=-1)
        # Extract upper triangle (exclude diagonal)
        upper_indices = np.triu_indices(17, k=1)
        feats.append(dists[upper_indices])
    return np.stack(feats)  # (T, 136)


def pool_temporal_features(seq_feats):
    """
    Pool (T, 136) to (272,) using mean + std.
    Mean captures average pose, std captures motion amplitude.
    """
    mean_feats = seq_feats.mean(axis=0)  # (136,)
    std_feats = seq_feats.std(axis=0)    # (136,)
    return np.concatenate([mean_feats, std_feats])  # (272,)


class PairwiseActionDataset(Dataset):
    def __init__(self, seqs, labels, window_len=120, mean=None, std=None):
        """
        seqs: list of (T, 17, 3) sequences
        labels: list of action labels
        """
        self.window_len = window_len
        self.data = []
        
        # Extract pairwise features and pool
        for seq, label in zip(seqs, labels):
            # Create sliding windows
            T = seq.shape[0]
            if T <= window_len:
                # Resample to window_len
                indices = np.linspace(0, T-1, window_len).astype(int)
                windows = [seq[indices]]
            else:
                # Sliding window with stride=40
                stride = 40
                windows = []
                for s in range(0, T-window_len+1, stride):
                    windows.append(seq[s:s+window_len])
                # Add last window if needed
                if (T - window_len) % stride != 0:
                    windows.append(seq[T-window_len:])
            
            # Process each window
            for win in windows:
                # Convert to pairwise features: (120, 136)
                pairwise = seq_to_pairwise_features(win)
                # Pool to (272,)
                pooled = pool_temporal_features(pairwise)
                self.data.append((pooled, label))
        
        # Compute normalization stats
        if mean is None or std is None:
            all_feats = np.stack([x[0] for x in self.data])
            self.mean = all_feats.mean(axis=0)
            self.std = all_feats.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std
        
        logger.info(f'  Created {len(self.data)} windows from {len(seqs)} sequences')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feats, label = self.data[idx]
        # Normalize
        feats = (feats - self.mean) / self.std
        return torch.from_numpy(feats).float(), label
