"""
L4核心实验2: 重复执行评分方差
目标: 验证评分规则的稳定性（同一动作多次执行的评分波动）
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from evaluators.scorer_v2 import QualityScorer
from rules.action_standards import H36M_ACTIONS

# 加载GT数据
GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

print("加载GT 3D数据 (所有subject)...")
gt_data = np.load('../dataset/data_3d_h36m.npz', allow_pickle=True)
gt_positions = gt_data['positions_3d'].item()

# 按 (subject, action) 组织数据
action_executions = {action: {} for action in H36M_ACTIONS}

for subject in [1, 5, 6, 7, 8, 9, 11]:
    subj_key = f'S{subject}'
    if subj_key not in gt_positions:
        continue
    
    for action_name, seq_32 in gt_positions[subj_key].items():
        action_base = action_name.split(' ')[0]
        if action_base not in H36M_ACTIONS:
            continue
        
        seq_17 = seq_32[:, GT_32_TO_17, :]
        seq_17 = seq_17 - seq_17[:, 0:1, :]
        
        if subject not in action_executions[action_base]:
            action_executions[action_base][subject] = []
        action_executions[action_base][subject].append(seq_17)

print(f"数据加载完成\n")

# 计算每个动作的重复执行方差
print(f"{'='*70}")
print("重复执行评分方差分析")
print(f"{'='*70}\n")
print(f"{'动作':<15} {'受试者数':<10} {'平均方差':<12} {'最大方差':<12} {'判决'}")
print(f"{'-'*70}")

all_variances = []
results = {}

for action in H36M_ACTIONS:
    scorer = QualityScorer(action)
    
    subject_variances = []
    subject_means = []
    
    for subject, executions in action_executions[action].items():
        if len(executions) < 2:
            continue
        
        scores = []
        for seq in executions:
            try:
                result = scorer.score(seq)
                scores.append(result['total'])
            except:
                continue
        
        if len(scores) >= 2:
            variance = np.var(scores)
            mean_score = np.mean(scores)
            subject_variances.append(variance)
            subject_means.append(mean_score)
    
    if subject_variances:
        avg_variance = np.mean(subject_variances)
        max_variance = np.max(subject_variances)
        std_score = np.sqrt(avg_variance)
        
        all_variances.append(avg_variance)
        
        # 判决标准: std < 3分为优秀, 3-5为良好, >5为需改进
        if std_score < 3.0:
            verdict = "✅ 优秀"
        elif std_score < 5.0:
            verdict = "⚠️  良好"
        else:
            verdict = "❌ 需改进"
        
        results[action] = {
            'num_subjects': len(subject_variances),
            'avg_variance': avg_variance,
            'std': std_score,
            'max_variance': max_variance,
            'verdict': verdict
        }
        
        print(f"{action:<15} {len(subject_variances):<10} {avg_variance:<12.2f} {max_variance:<12.2f} {verdict}")

# 总体统计
print(f"\n{'总体统计':<15} {len(all_variances):<10} {np.mean(all_variances):<12.2f} {np.max(all_variances):<12.2f}")

# 需要改进的动作
print(f"\n【需要改进的动作】(std > 5分)")
for action, stats in results.items():
    if stats['std'] > 5.0:
        print(f"  {action}: std={stats['std']:.2f}, 建议: 增加平滑或调整阈值")

# 保存结果
output_file = 'results_repeatability.npz'
np.savez(output_file, results=results)
print(f"\n结果已保存到: {output_file}")

print(f"\n{'='*70}")
print(f"实验完成！")
print(f"{'='*70}\n")
