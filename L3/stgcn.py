"""
L3 ST-GCN - Fixed version with:
- Correct graph conv dimension handling (B,C,T,V throughout)
- Smaller model (5 layers, 32/64/128 channels) matching data size
- Lower dropout (0.1) to allow gradient flow
- Correct data_bn application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

H36M_ACTIONS = [
    'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
    'Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown',
    'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether'
]

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]


def build_h36m_adj(num_nodes=17):
    """Symmetric normalized adjacency matrix for H36M 17-joint skeleton."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in H36M_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    d = adj.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, np.power(d, -0.5), 0.0)
    D = np.diag(d_inv_sqrt)
    return (D @ adj @ D).astype(np.float32)


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution.
    Input/output: (B, C, T, V) - channels first, consistent throughout.
    Linear transform via 1x1 conv, then aggregate via adjacency.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 conv: transform features per-node per-timestep
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, adj):
        # x: (B, C_in, T, V)
        y = self.conv(x)                              # (B, C_out, T, V)
        # Aggregate neighbors: sum over source nodes weighted by adj
        # y: (B, C, T, V), adj: (V, V) -> (B, C, T, V)
        y = torch.einsum('bctv,vw->bctw', y, adj)    # (B, C_out, T, V)
        return self.bn(y)


class TemporalConv(nn.Module):
    """Temporal convolution along time axis."""
    def __init__(self, channels, kernel_size=5, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad, 0)
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    """ST-GCN block: SpatialGCN + TemporalConv + residual."""
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dropout=0.1):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch)
        self.tcn = TemporalConv(out_ch, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x, adj):
        res = self.residual(x)
        out = self.relu(self.gcn(x, adj))
        out = self.drop(self.tcn(out))
        # Align time if stride changed it
        if out.shape[2] != res.shape[2]:
            res = res[:, :, :out.shape[2], :]
        return self.relu(out + res)


class STGCN(nn.Module):
    """
    Path-2: Smaller 5-layer ST-GCN (32/64/128 channels, kernel=5).
    ~0.4M params for ~1000 training samples.
    
    Input: (B, T, V, C) or (B, C, T, V)
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=15, num_nodes=17, in_channels=3, dropout=0.1):
        super().__init__()
        adj = build_h36m_adj(num_nodes)
        self.register_buffer('adj', torch.from_numpy(adj))

        # Input BN: normalize over (C*V) dimension, applied on (B, C*V, T)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 5 blocks: 32x2 | 64x2 (stride=2 at block 3) | 128x1  (~0.16M params, matched to 9K samples)
        cfg = [
            (in_channels, 32, 1),   # block 1
            (32, 32, 1),            # block 2
            (32, 64, 2),            # block 3: stride=2
            (64, 64, 1),            # block 4
            (64, 128, 1),           # block 5
        ]
        self.blocks = nn.ModuleList([
            STGCNBlock(inc, outc, kernel_size=5, stride=s, dropout=dropout)
            for inc, outc, s in cfg
        ])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """x: (B, T, V, C) -> logits (B, num_classes)"""
        # Normalize input format to (B, C, T, V)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, T, V)
        B, C, T, V = x.shape

        # data_bn expects (B, C*V, T)
        # x is (B, C, T, V) -> rearrange to (B, C*V, T)
        xbn = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)  # (B, C*V, T)
        xbn = self.data_bn(xbn)
        x = xbn.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()   # (B, C, T, V)

        for blk in self.blocks:
            x = blk(x, self.adj)

        x = self.global_pool(x).view(B, -1)   # (B, 128)
        return self.fc(x)
