"""
在GT数据上测试评分系统
验证L4规范性评分模块的有效性
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from evaluators.quality_scorer import QualityScorer, batch_score
from rules.action_standards import H36M_ACTIONS


# Human3.6M GT数据映射 (32 -> 17关节)
GT_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]


def load_h36m_gt(data_path='../dataset/data_3d_h36m.npz', subjects=[9, 11], max_seq_per_action=3):
    """
    加载Human3.6M GT数据
    
    Args:
        data_path: 数据路径
        subjects: 测试subject列表
        max_seq_per_action: 每个动作最多加载多少个序列
    
    Returns:
        (sequences, labels, action_names)
    """
    data = np.load(data_path, allow_pickle=True)
    positions_3d = data['positions_3d'].item()
    
    sequences = []
    labels = []
    
    action_counts = {action: 0 for action in H36M_ACTIONS}
    
    for subject in subjects:
        subj_key = f'S{subject}'
        if subj_key not in positions_3d:
            continue
        
        for action_name, seq_32 in positions_3d[subj_key].items():
            action_base = action_name.split(' ')[0]
            
            if action_base not in H36M_ACTIONS:
                continue
            
            # 限制每个动作的序列数
            if action_counts[action_base] >= max_seq_per_action:
                continue
            
            # 映射到17关节
            seq_17 = seq_32[:, GT_32_TO_17, :]
            
            # Hip关节归零（相对坐标）
            seq_17 = seq_17 - seq_17[:, 0:1, :]
            
            label = H36M_ACTIONS.index(action_base)
            
            sequences.append(seq_17)
            labels.append(label)
            action_counts[action_base] += 1
    
    print(f'加载了 {len(sequences)} 个序列')
    print(f'动作分布: {action_counts}')
    
    return sequences, labels, H36M_ACTIONS


def test_single_action(action_name='Walking', subject=9):
    """测试单个动作的评分"""
    print(f"\n{'='*60}")
    print(f"测试单个动作: {action_name}")
    print(f"{'='*60}\n")
    
    sequences, labels, action_names = load_h36m_gt(
        subjects=[subject],
        max_seq_per_action=1
    )
    
    # 找到目标动作
    target_label = H36M_ACTIONS.index(action_name)
    for seq, label in zip(sequences, labels):
        if label == target_label:
            scorer = QualityScorer(action_name)
            result = scorer.score(seq, verbose=True)
            return result
    
    print(f"未找到动作: {action_name}")
    return None


def test_all_actions():
    """测试所有15个动作"""
    print(f"\n{'='*60}")
    print(f"测试所有动作类别")
    print(f"{'='*60}\n")
    
    sequences, labels, action_names = load_h36m_gt(
        subjects=[9, 11],
        max_seq_per_action=2
    )
    
    print(f"\n开始批量评分...")
    results = batch_score(sequences, labels, action_names, verbose=False)
    
    # 统计结果
    print(f"\n{'='*60}")
    print(f"评分统计")
    print(f"{'='*60}\n")
    
    action_scores = {action: [] for action in H36M_ACTIONS}
    for result in results:
        action_scores[result['action']].append(result['total'])
    
    print(f"{'动作':<20} {'样本数':<8} {'平均分':<10} {'标准差':<10}")
    print(f"{'-'*60}")
    
    for action in H36M_ACTIONS:
        scores = action_scores[action]
        if len(scores) > 0:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{action:<20} {len(scores):<8} {mean_score:<10.1f} {std_score:<10.2f}")
    
    # 总体统计
    all_scores = [r['total'] for r in results]
    print(f"\n{'总体统计':<20} {len(all_scores):<8} {np.mean(all_scores):<10.1f} {np.std(all_scores):<10.2f}")
    
    # 维度统计
    print(f"\n各维度平均分:")
    print(f"  关节角度:   {np.mean([r['angle'] for r in results]):.1f}/30")
    print(f"  动作幅度:   {np.mean([r['amplitude'] for r in results]):.1f}/25")
    print(f"  对称性:     {np.mean([r['symmetry'] for r in results]):.1f}/20")
    print(f"  节奏:       {np.mean([r['rhythm'] for r in results]):.1f}/15")
    print(f"  稳定性:     {np.mean([r['stability'] for r in results]):.1f}/10")
    
    return results


def test_score_sensitivity():
    """测试评分系统对姿态误差的敏感性"""
    print(f"\n{'='*60}")
    print(f"测试评分对姿态误差的敏感性")
    print(f"{'='*60}\n")
    
    # 加载一个干净的GT序列
    sequences, labels, action_names = load_h36m_gt(subjects=[9], max_seq_per_action=1)
    
    if len(sequences) == 0:
        print("无法加载测试序列")
        return
    
    seq_clean = sequences[0]
    label = labels[0]
    action_name = action_names[label]
    
    scorer = QualityScorer(action_name)
    
    # 原始得分
    result_clean = scorer.score(seq_clean)
    print(f"原始GT得分: {result_clean['total']:.1f}/100\n")
    
    # 添加不同级别的高斯噪声
    noise_levels = [0.01, 0.03, 0.05, 0.10]  # 米
    
    print(f"{'噪声水平(m)':<15} {'总分':<10} {'分数下降':<10}")
    print(f"{'-'*40}")
    
    for noise_std in noise_levels:
        seq_noisy = seq_clean + np.random.randn(*seq_clean.shape) * noise_std
        result_noisy = scorer.score(seq_noisy)
        
        score_drop = result_clean['total'] - result_noisy['total']
        print(f"{noise_std:<15.2f} {result_noisy['total']:<10.1f} {score_drop:<10.1f}")
    
    print(f"\n说明: L2的P1=0.048m对应噪声水平约为0.05m")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='L4评分系统测试')
    parser.add_argument('--mode', choices=['single', 'all', 'sensitivity'], default='all',
                       help='测试模式')
    parser.add_argument('--action', default='Walking', help='单动作测试时的动作名称')
    parser.add_argument('--subject', type=int, default=9, help='Subject ID')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_single_action(args.action, args.subject)
    elif args.mode == 'all':
        test_all_actions()
    elif args.mode == 'sensitivity':
        test_score_sensitivity()
