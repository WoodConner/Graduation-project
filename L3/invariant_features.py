"""
Pairwise distance features: invariant to joint order and absolute scale.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

def seq_to_pairwise_distances(seq):
    """
    Convert (T, 17, 3) to (T, 136) pairwise distance features.
    136 = 17*16/2 (upper triangle of distance matrix)
    """
    T, J, _ = seq.shape
    feats = []
    for t in range(T):
        dists = np.linalg.norm(seq[t][:, None] - seq[t][None, :], axis=-1)
        triu_idx = np.triu_indices(J, k=1)
        feats.append(dists[triu_idx])
    return np.stack(feats)  # (T, 136)


class InvariantDataset(Dataset):
    def __init__(self, seqs, labels, window=120, augment=False):
        self.data, self.labels = [], []
        for seq, label in zip(seqs, labels):
            T = seq.shape[0]
            if T < window:
                continue
            feat = seq_to_pairwise_distances(seq)  # (T, 136)
            
            if augment:
                for start in range(0, T - window + 1, window // 4):
                    self.data.append(feat[start:start + window])
                    self.labels.append(label)
            else:
                mid = (T - window) // 2
                self.data.append(feat[mid:mid + window])
                self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()  # (120, 136)
        y = self.labels[idx]
        return x, y
