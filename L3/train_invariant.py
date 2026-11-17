#!/usr/bin/env python3
"""L3: Invariant feature-based action recognition"""
import argparse, logging, random, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent))
from invariant_features import InvariantDataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

H36M_ACTIONS = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
                'Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
                'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']

class MLP_Classifier(nn.Module):
    def __init__(self, input_dim=120*136, hidden=512, num_classes=15, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_one_seed(train_seqs, train_labels, val_seqs, val_labels, args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_ds = InvariantDataset(train_seqs, train_labels, window=args.window, augment=True)
    val_ds = InvariantDataset(val_seqs, val_labels, window=args.window, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    model = MLP_Classifier(input_dim=args.window * 136, hidden=args.hidden, 
                           num_classes=15, dropout=args.dropout).cuda()
    logger.info(f'  Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, no_improve = 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f'  E {epoch}/{args.epochs} loss={total_loss/len(train_loader):.4f} '
                       f'val={val_acc:.4f} best={best_acc:.4f} (no_improve={no_improve})')
        if no_improve >= args.early_stop_patience:
            logger.info(f'  Early stop at epoch {epoch}')
            break
    return best_acc

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['G1', 'G2'], required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--window', type=int, default=120)
    p.add_argument('--early_stop_patience', type=int, default=10)
    p.add_argument('--l2_path', default='L3/l2_fixed.npz')
    p.add_argument('--gt_path', default='dataset/data_3d_h36m.npz')
    args = p.parse_args()
    logger.info(f'[{args.group}] Loading data...')
    gt_data = np.load(args.gt_path, allow_pickle=True)
    gt_dict = gt_data['positions_3d'].item()
    H36M_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
    train_seqs, train_labels = [], []
    for subj in [1, 5, 6, 7, 8]:
        key = f'S{subj}'
        if key not in gt_dict:
            continue
        for action_name, seq_32 in gt_dict[key].items():
            base = action_name.split(' ')[0]
            if base not in H36M_ACTIONS:
                continue
            label = H36M_ACTIONS.index(base)
            seq_17 = seq_32[:, H36M_32_TO_17, :]
            train_seqs.append(seq_17)
            train_labels.append(label)
    logger.info(f'  Train: {len(train_seqs)} sequences')
    if args.group == 'G1':
        val_seqs, val_labels = [], []
        for subj in [9, 11]:
            key = f'S{subj}'
            if key not in gt_dict:
                continue
            for action_name, seq_32 in gt_dict[key].items():
                base = action_name.split(' ')[0]
                if base not in H36M_ACTIONS:
                    continue
                label = H36M_ACTIONS.index(base)
                seq_17 = seq_32[:, H36M_32_TO_17, :]
                val_seqs.append(seq_17)
                val_labels.append(label)
        logger.info(f'  Val: {len(val_seqs)} sequences (GT)')
    else:
        l2_data = np.load(args.l2_path, allow_pickle=True)
        val_seqs = list(l2_data['val_seqs'])
        val_labels = list(l2_data['val_labels'])
        logger.info(f'  Val: {len(val_seqs)} sequences (L2)')
    logger.info(f'[{args.group}] Training {args.seeds} seeds...')
    accs = []
    for i in range(args.seeds):
        logger.info(f'[{args.group}] Seed {i+1}/{args.seeds}')
        acc = train_one_seed(train_seqs, train_labels, val_seqs, val_labels, args, seed=i)
        accs.append(acc)
        logger.info(f'[{args.group}] Seed {i} = {acc:.4f}')
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f'{args.group}: {mean_acc:.4f}+/-{std_acc:.4f}')

if __name__ == '__main__':
    main()
