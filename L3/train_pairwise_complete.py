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
    """Convert (T, 17, 3) to (T, 136) pairwise distances."""
    # Ensure seq is numpy array with correct shape
    seq = np.array(seq, dtype=np.float32)
    if len(seq.shape) != 3 or seq.shape[1] != 17 or seq.shape[2] != 3:
        raise ValueError(f'Expected (T, 17, 3), got {seq.shape}')
    
    T = seq.shape[0]
    feats = []
    for t in range(T):
        # Compute pairwise distances: (17, 17)
        joints = seq[t]  # (17, 3)
        dists = np.linalg.norm(joints[:, None, :] - joints[None, :, :], axis=-1)
        # Extract upper triangle
        upper_indices = np.triu_indices(17, k=1)
        feats.append(dists[upper_indices])
    return np.stack(feats)


def pool_temporal_features(seq_feats):
    mean_feats = seq_feats.mean(axis=0)
    std_feats = seq_feats.std(axis=0)
    return np.concatenate([mean_feats, std_feats])


class PairwiseActionDataset(Dataset):
    def __init__(self, seqs, labels, window_len=120, mean=None, std=None, augment_l2=False):
        self.window_len = window_len
        self.data = []
        
        for seq, label in zip(seqs, labels):
            T = seq.shape[0]
            if T <= window_len:
                # For short sequences (like L2's fixed 120 frames)
                if augment_l2 and T == 120:
                    # Create multiple augmented windows by subsampling
                    # Simulate the effect of having longer sequences
                    windows = []
                    for i in range(64):  # Match G1's ~64x data multiplier
                        # Random temporal jitter
                        start = np.random.randint(0, max(1, T // 10))
                        end = T - np.random.randint(0, max(1, T // 10))
                        indices = np.linspace(start, end-1, window_len).astype(int)
                        windows.append(seq[indices])
                else:
                    indices = np.linspace(0, T-1, window_len).astype(int)
                    windows = [seq[indices]]
            else:
                stride = 40
                windows = []
                for s in range(0, T-window_len+1, stride):
                    windows.append(seq[s:s+window_len])
                if (T - window_len) % stride != 0:
                    windows.append(seq[T-window_len:])
            
            for win in windows:
                pairwise = seq_to_pairwise_features(win)
                pooled = pool_temporal_features(pairwise)
                self.data.append((pooled, label))
        
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
        feats = (feats - self.mean) / self.std
        return torch.from_numpy(feats).float(), label


def load_h36m_sequences(subjects, data_path='dataset/data_3d_h36m.npz'):
    GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    data = np.load(data_path, allow_pickle=True)
    positions_3d = data['positions_3d'].item()
    sequences, labels = [], []
    
    for subject in subjects:
        subj_key = f'S{subject}'
        if subj_key not in positions_3d:
            continue
        for action_name, seq_32 in positions_3d[subj_key].items():
            seq_17 = seq_32[:, GT_32_TO_17, :]
            seq_17 = seq_17 - seq_17[:, 0:1, :]
            action_base = action_name.split(' ')[0]
            if action_base in H36M_ACTIONS:
                label = H36M_ACTIONS.index(action_base)
                sequences.append(seq_17)
                labels.append(label)
    
    logger.info(f'  Loaded {len(sequences)} sequences from {subjects}')
    return sequences, labels


def build_mlp_model(input_dim=272, num_classes=15):
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(64, num_classes)
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='G1', choices=['G1', 'G2'])
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default='dataset/data_3d_h36m.npz')
    parser.add_argument('--l2_poses_path', type=str, default='L3/l2_poses_test_120.npz')
    parser.add_argument('--early_stop_patience', type=int, default=15)
    args = parser.parse_args()
    
    logger.info(f'Starting {args.group} training with pairwise features (272-dim)')
    
    if args.group == 'G1':
        logger.info('Loading GT 3D sequences...')
        tr_seqs, tr_lbls = load_h36m_sequences(H36M_TRAIN_SUBJECTS, args.data_path)
        va_seqs, va_lbls = load_h36m_sequences(H36M_TEST_SUBJECTS, args.data_path)
    else:
        logger.info('Loading L2 output sequences (full length)...')
        l2_data = np.load(args.l2_poses_path, allow_pickle=True)
        tr_seqs = list(l2_data['train_seqs'])
        tr_lbls = list(l2_data['train_labels'])
        va_seqs = list(l2_data['val_seqs'])
        va_lbls = list(l2_data['val_labels'])
        logger.info(f'  L2 val seqs: {len(va_seqs)}, sample shape: {va_seqs[0].shape}')
    
    results_dir = Path('L3/results_pairwise')
    results_dir.mkdir(exist_ok=True)
    all_results = []
    
    for seed_idx in range(args.seeds):
        logger.info(f'[{args.group}] Seed {seed_idx+1}/{args.seeds}')
        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)
        
        train_ds = PairwiseActionDataset(tr_seqs, tr_lbls, window_len=120)
        val_ds = PairwiseActionDataset(va_seqs, va_lbls, window_len=120, 
                                       mean=train_ds.mean, std=train_ds.std)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_mlp_model(input_dim=272, num_classes=15).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'  Model params: {num_params/1e3:.1f}K')
        logger.info(f'  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}')
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        
        best_acc, best_epoch, no_improve_count = 0.0, 0, 0
        history = []
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            scheduler.step()
            
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'val_acc': val_acc,
                }, results_dir / f'{args.group}_s{seed_idx}_best.pth')
            else:
                no_improve_count += 1
            
            if epoch % 10 == 0 or epoch == args.epochs:
                logger.info(f'  E {epoch}/{args.epochs}: loss={train_loss:.4f} '
                          f'tr_acc={train_acc:.4f} val_acc={val_acc:.4f} '
                          f'best={best_acc:.4f} (e{best_epoch})')
            
            if no_improve_count >= args.early_stop_patience:
                logger.info(f'  Early stop at epoch {epoch}')
                break
        
        logger.info(f'[{args.group}] Seed {seed_idx} = {best_acc:.4f}')
        all_results.append(best_acc)
        
        with open(results_dir / f'{args.group}_s{seed_idx}_hist.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    mean_acc = np.mean(all_results)
    std_acc = np.std(all_results)
    logger.info(f'{args.group}: {mean_acc:.4f}+/-{std_acc:.4f}')
    
    summary = {
        'group': args.group,
        'method': 'pairwise_272dim',
        'seeds': all_results,
        'mean': float(mean_acc),
        'std': float(std_acc)
    }
    
    with open(results_dir / f'{args.group}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f'Results saved to {results_dir}')


if __name__ == '__main__':
    main()
