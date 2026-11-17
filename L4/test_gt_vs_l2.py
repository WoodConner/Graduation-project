"""
L4核心实验1: GT 3D vs L2输出的评分对比
目标: 将"P1=48mm"翻译成"评分偏差X分/排序一致性rho=Y"
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from scipy.stats import spearmanr
from evaluators.scorer_v2 import QualityScorer
from rules.action_standards import H36M_ACTIONS

# 加载GT数据
GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

print("加载GT 3D数据 (S9, S11)...")
gt_data = np.load('../dataset/data_3d_h36m.npz', allow_pickle=True)
gt_positions = gt_data['positions_3d'].item()

gt_seqs = []
gt_labels = []
gt_names = []

for subject in [9, 11]:
    subj_key = f'S{subject}'
    if subj_key not in gt_positions:
        continue
    for action_name, seq_32 in gt_positions[subj_key].items():
        action_base = action_name.split(' ')[0]
        if action_base not in H36M_ACTIONS:
            continue
        seq_17 = seq_32[:, GT_32_TO_17, :]
        seq_17 = seq_17 - seq_17[:, 0:1, :]  # Hip归零
        gt_seqs.append(seq_17)
        gt_labels.append(H36M_ACTIONS.index(action_base))
        gt_names.append(action_base)

print(f"  加载了 {len(gt_seqs)} 个GT序列")

# 加载L2数据
print("\n加载L2输出数据 (S9, S11)...")
try:
    l2_data = np.load('../L3/l2_poses_full.npz', allow_pickle=True)
    l2_seqs_raw = l2_data['val_seqs']
    l2_labels = l2_data['val_labels']
    print(f"  加载了 {len(l2_seqs_raw)} 个L2序列")
    
    # L2序列需要Hip归零
    l2_seqs = []
    for seq in l2_seqs_raw:
        seq_centered = seq - seq[:, 0:1, :]
        l2_seqs.append(seq_centered)
    
except FileNotFoundError:
    print("  ⚠️  L3/l2_poses_full.npz 不存在，使用L3/l2_poses_test_120.npz作为替代")
    l2_data = np.load('../L3/l2_poses_test_120.npz', allow_pickle=True)
    l2_seqs_raw = l2_data['val_seqs']
    l2_labels = l2_data['val_labels']
    print(f"  加载了 {len(l2_seqs_raw)} 个L2序列 (120帧版本)")
    l2_seqs = [seq - seq[:, 0:1, :] for seq in l2_seqs_raw]

# 检查数据对齐
print(f"\n数据对齐检查:")
print(f"  GT序列数: {len(gt_seqs)}, L2序列数: {len(l2_seqs)}")

if len(gt_seqs) != len(l2_seqs):
    print(f"  ⚠️  序列数不匹配，取较小值")
    min_len = min(len(gt_seqs), len(l2_seqs))
    gt_seqs = gt_seqs[:min_len]
    l2_seqs = l2_seqs[:min_len]
    gt_labels = gt_labels[:min_len]
    gt_names = gt_names[:min_len]

# 对每个序列评分
print(f"\n开始评分...")
results = []

for i, (gt_seq, l2_seq, label, action_name) in enumerate(zip(gt_seqs, l2_seqs, gt_labels, gt_names)):
    if i % 10 == 0:
        print(f"  进度: {i}/{len(gt_seqs)}")
    
    # 截断或填充到相同长度
    min_len = min(len(gt_seq), len(l2_seq))
    gt_seq_trimmed = gt_seq[:min_len]
    l2_seq_trimmed = l2_seq[:min_len]
    
    scorer = QualityScorer(action_name)
    
    try:
        gt_result = scorer.score(gt_seq_trimmed)
        l2_result = scorer.score(l2_seq_trimmed)
        
        results.append({
            'action': action_name,
            'gt_total': gt_result['total'],
            'l2_total': l2_result['total'],
            'delta': abs(gt_result['total'] - l2_result['total']),
            'gt_angle': gt_result['angle'],
            'l2_angle': l2_result['angle'],
            'gt_symmetry': gt_result['symmetry'],
            'l2_symmetry': l2_result['symmetry'],
        })
    except Exception as e:
        print(f"  ⚠️  序列{i} ({action_name}) 评分失败: {e}")
        continue

print(f"\n成功评分 {len(results)} 个序列")

# ==================== 核心指标计算 ====================

print(f"\n{'='*70}")
print("L4核心实验结果: GT vs L2评分对比")
print(f"{'='*70}\n")

# 1. 平均绝对偏差
deltas = [r['delta'] for r in results]
mean_delta = np.mean(deltas)
median_delta = np.median(deltas)
max_delta = np.max(deltas)
std_delta = np.std(deltas)

print(f"【评分偏差统计】")
print(f"  平均绝对偏差:  {mean_delta:.2f} 分")
print(f"  中位数偏差:    {median_delta:.2f} 分")
print(f"  最大偏差:      {max_delta:.2f} 分")
print(f"  标准差:        {std_delta:.2f} 分")

# 2. Spearman排名相关性
gt_scores = np.array([r['gt_total'] for r in results])
l2_scores = np.array([r['l2_total'] for r in results])
rho, p_value = spearmanr(gt_scores, l2_scores)

print(f"\n【排名一致性】")
print(f"  Spearman ρ:    {rho:.3f}")
print(f"  p-value:       {p_value:.4f}")

# 3. 判决
print(f"\n【L4判决】")
if mean_delta < 5.0 and rho > 0.8:
    verdict = "✅ L2输出可直接用于L4 - 误差影响可忽略"
elif mean_delta < 10.0 and rho > 0.7:
    verdict = "⚠️  L2输出需要轻微校准 - 排序基本一致但绝对值有偏差"
elif rho > 0.6:
    verdict = "⚠️  L2输出需要显著校准 - 排序部分一致"
else:
    verdict = "❌ L2输出不可用于L4 - 需要修复L2或使用GT"

print(f"  {verdict}")

# 4. 逐动作偏差
print(f"\n【逐动作偏差分析】")
print(f"{'动作':<15} {'样本数':<8} {'平均偏差':<12} {'GT均分':<10} {'L2均分':<10}")
print(f"{'-'*70}")

action_stats = {}
for action in H36M_ACTIONS:
    action_results = [r for r in results if r['action'] == action]
    if not action_results:
        continue
    
    action_deltas = [r['delta'] for r in action_results]
    action_gt = [r['gt_total'] for r in action_results]
    action_l2 = [r['l2_total'] for r in action_results]
    
    action_stats[action] = {
        'count': len(action_results),
        'mean_delta': np.mean(action_deltas),
        'mean_gt': np.mean(action_gt),
        'mean_l2': np.mean(action_l2),
    }
    
    print(f"{action:<15} {len(action_results):<8} {np.mean(action_deltas):<12.1f} {np.mean(action_gt):<10.1f} {np.mean(action_l2):<10.1f}")

# 5. 线性校准建议
if mean_delta > 5.0:
    print(f"\n【线性校准分析】")
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(l2_scores.reshape(-1, 1), gt_scores)
    l2_calibrated = reg.predict(l2_scores.reshape(-1, 1))
    
    calibrated_deltas = np.abs(gt_scores - l2_calibrated)
    print(f"  校准前平均偏差: {mean_delta:.2f} 分")
    print(f"  校准后平均偏差: {np.mean(calibrated_deltas):.2f} 分")
    print(f"  校准参数: L2_calibrated = {reg.coef_[0]:.3f} × L2 + {reg.intercept_:.3f}")

# 6. 保存结果
output_file = 'results_gt_vs_l2.npz'
np.savez(output_file,
         results=results,
         mean_delta=mean_delta,
         spearman_rho=rho,
         verdict=verdict,
         action_stats=action_stats)
print(f"\n结果已保存到: {output_file}")

print(f"\n{'='*70}")
print(f"实验完成！")
print(f"{'='*70}\n")
