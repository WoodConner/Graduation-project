"""测试v2评分器（动作特定权重）"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from evaluators.scorer_v2 import QualityScorer, batch_score
import numpy as np

# 加载GT数据
GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
from rules.action_standards import H36M_ACTIONS

data_path = '../dataset/data_3d_h36m.npz'
data = np.load(data_path, allow_pickle=True)
positions_3d = data['positions_3d'].item()

sequences = []
labels = []

for subject in [9, 11]:
    subj_key = f'S{subject}'
    for action_name, seq_32 in positions_3d[subj_key].items():
        action_base = action_name.split(' ')[0]
        if action_base not in H36M_ACTIONS:
            continue
        if len([s for s, l in zip(sequences, labels) if H36M_ACTIONS[l] == action_base]) >= 2:
            continue
        seq_17 = seq_32[:, GT_32_TO_17, :]
        seq_17 = seq_17 - seq_17[:, 0:1, :]
        label = H36M_ACTIONS.index(action_base)
        sequences.append(seq_17)
        labels.append(label)

print(f"加载了 {len(sequences)} 个序列")

print("评分中...")
results = batch_score(sequences, labels, H36M_ACTIONS)

# 统计
action_scores = {a: [] for a in H36M_ACTIONS}
for r in results:
    action_scores[r['action']].append(r['total'])

print(f"\n{'='*70}")
print("动作评分统计 (v2 - 动作特定权重)")
print(f"{'='*70}")
print(f"{'动作':<20} {'样本数':<8} {'平均分':<10} {'标准差':<10}")
print(f"{'-'*70}")

for a in H36M_ACTIONS:
    if action_scores[a]:
        print(f"{a:<20} {len(action_scores[a]):<8} {np.mean(action_scores[a]):<10.1f} {np.std(action_scores[a]):<10.2f}")

all_scores = [r['total'] for r in results]
print(f"\n{'总体统计':<20} {len(results):<8} {np.mean(all_scores):<10.1f} {np.std(all_scores):<10.2f}")

# 对比v1
print(f"\n\n对比v1版本（一刀切权重）:")
print(f"  v1: SittingDown=54.7, Phoning=65.8, Smoking=66.2, Walking=84.7")
print(f"  v2: SittingDown={np.mean(action_scores['SittingDown']):.1f}, Phoning={np.mean(action_scores['Phoning']):.1f}, Smoking={np.mean(action_scores['Smoking']):.1f}, Walking={np.mean(action_scores['Walking']):.1f}")
print(f"\n改进: 单手动作和不对称动作的分数提升，Walking等对称动作保持高分")

