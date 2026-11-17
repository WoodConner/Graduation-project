"""
L3 Dataset with sliding window augmentation.
Train: 150 seqs -> ~900-1500 windows (stride=40, 50% overlap)
Test: 59 seqs -> 59 center windows (no overlap)
"""
import numpy as np, torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

H36M_ACTIONS = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting','SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
H36M_FLIP_PAIRS = [(1,4),(2,5),(3,6),(11,14),(12,15),(13,16)]
H36M_TRAIN_SUBJECTS = [1,5,6,7,8]
H36M_TEST_SUBJECTS = [9,11]


def create_sliding_windows(sequences, labels, window_len=120, stride=40):
    windows, window_labels = [], []
    for seq, label in zip(sequences, labels):
        T = seq.shape[0]
        if T < window_len:
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, window_len)
            padded = np.zeros((window_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
            for v in range(seq.shape[1]):
                for c in range(seq.shape[2]):
                    padded[:, v, c] = interp1d(old_t, seq[:, v, c], kind='linear')(new_t)
            windows.append(padded); window_labels.append(label)
        else:
            for start in range(0, T - window_len + 1, stride):
                windows.append(seq[start:start+window_len].copy()); window_labels.append(label)
            if (T - window_len) % stride != 0:
                windows.append(seq[T - window_len:].copy()); window_labels.append(label)
    return windows, window_labels


def create_test_windows(sequences, labels, window_len=120):
    windows, window_labels = [], []
    for seq, label in zip(sequences, labels):
        T = seq.shape[0]
        if T < window_len:
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, window_len)
            padded = np.zeros((window_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
            for v in range(seq.shape[1]):
                for c in range(seq.shape[2]):
                    padded[:, v, c] = interp1d(old_t, seq[:, v, c], kind='linear')(new_t)
            windows.append(padded); window_labels.append(label)
        else:
            center = T // 2
            start = max(0, center - window_len // 2)
            windows.append(seq[start:start+window_len].copy()); window_labels.append(label)
    return windows, window_labels


def center_root(seq):
    return seq - seq[:, 0:1, :]


def flip_skeleton(seq, flip_pairs=H36M_FLIP_PAIRS):
    seq = seq.copy(); seq[:, :, 0] = -seq[:, :, 0]
    for l, r in flip_pairs:
        seq[:, [l, r], :] = seq[:, [r, l], :]
    return seq


def random_rotate_y(seq, max_angle_deg=30):
    angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
    c, s = np.cos(angle), np.sin(angle)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return seq @ Ry.T


def random_scale(seq, scale_range=(0.9, 1.1)):
    return seq * np.random.uniform(*scale_range)


class H36MActionDataset(Dataset):
    def __init__(self, sequences, labels, mean=None, std=None,
                 window_len=120, stride=40, is_train=False,
                 use_sliding_window=True, use_augmentation=True,
                 aug_rotate=15, aug_scale=0.05):
        if use_sliding_window:
            if is_train:
                self.sequences, self.labels = create_sliding_windows(sequences, labels, window_len, stride)
            else:
                self.sequences, self.labels = create_test_windows(sequences, labels, window_len)
        else:
            self.sequences = []
            self.labels = list(labels)
            for seq in sequences:
                T = seq.shape[0]
                if T < window_len:
                    old_t = np.linspace(0, 1, T); new_t = np.linspace(0, 1, window_len)
                    padded = np.zeros((window_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
                    for v in range(seq.shape[1]):
                        for c in range(seq.shape[2]):
                            padded[:, v, c] = interp1d(old_t, seq[:, v, c], kind='linear')(new_t)
                    self.sequences.append(padded)
                else:
                    self.sequences.append(seq[:window_len].copy())
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.aug_rotate = aug_rotate
        self.aug_scale = aug_scale
        if mean is not None:
            self.mean = mean; self.std = std
        else:
            all_poses = np.concatenate([center_root(s) for s in self.sequences], axis=0)
            self.mean = all_poses.mean(axis=0, keepdims=True)
            self.std = all_poses.std(axis=0, keepdims=True) + 1e-8

    def set_stats(self, mean, std):
        self.mean = mean; self.std = std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy().astype(np.float32)
        seq = center_root(seq)
        if self.is_train and self.use_augmentation:
            seq = random_rotate_y(seq, self.aug_rotate)
            seq = random_scale(seq, (1.0 - self.aug_scale, 1.0 + self.aug_scale))
            if np.random.rand() < 0.5:
                seq = flip_skeleton(seq)
        seq = (seq - self.mean) / self.std
        return torch.from_numpy(seq), int(self.labels[idx])


def create_vote_windows(sequences, labels, window_len=120, n_windows=3):
    """
    For test set: take n_windows evenly spaced windows per sequence, then vote.
    Returns: list of (windows, label) where windows is a list of arrays.
    """
    result = []  # list of (list_of_windows, label)
    for seq, label in zip(sequences, labels):
        T = seq.shape[0]
        if T < window_len:
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, window_len)
            padded = np.zeros((window_len, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
            for v in range(seq.shape[1]):
                for c in range(seq.shape[2]):
                    padded[:, v, c] = interp1d(old_t, seq[:, v, c], kind='linear')(new_t)
            result.append(([padded], label))
        else:
            windows = []
            # n_windows evenly spaced start positions
            max_start = T - window_len
            for i in range(n_windows):
                start = int(max_start * i / max(n_windows - 1, 1))
                windows.append(seq[start:start+window_len].copy())
            result.append((windows, label))
    return result
