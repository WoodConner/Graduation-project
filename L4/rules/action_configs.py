"""
动作特定的评分配置
为每个动作定义专属的权重和阈值
"""
from typing import Dict

# 对称性权重（0-1）：0表示完全不要求对称，1表示必须对称
SYMMETRY_WEIGHTS = {
    'Directions': 0.4,      # 指向动作可能单手
    'Discussion': 0.5,      # 讨论时手势可能不对称
    'Eating': 0.2,          # 单手吃饭
    'Greeting': 0.4,        # 挥手可能单手
    'Phoning': 0.1,         # 单手打电话
    'Photo': 0.6,           # 拍照双手持机
    'Posing': 0.8,          # 摆姿势要求对称
    'Purchases': 0.4,       # 购物可能单手拿物
    'Sitting': 0.7,         # 坐姿应对称
    'SittingDown': 0.2,     # 坐下过程本身不对称
    'Smoking': 0.1,         # 单手抽烟
    'Waiting': 0.7,         # 等待时应对称站立
    'WalkDog': 0.5,         # 遛狗时牵绳手不对称
    'Walking': 0.9,         # 正常步态应高度对称
    'WalkTogether': 0.8,    # 一起走应对称
}

# 节奏权重（0-1）：0表示静态动作，1表示需要周期性节奏
RHYTHM_WEIGHTS = {
    'Directions': 0.3,
    'Discussion': 0.4,
    'Eating': 0.5,
    'Greeting': 0.7,
    'Phoning': 0.3,
    'Photo': 0.2,
    'Posing': 0.1,          # 摆姿势基本静态
    'Purchases': 0.4,
    'Sitting': 0.1,         # 静态坐姿
    'SittingDown': 0.8,     # 坐下需要流畅过渡
    'Smoking': 0.5,
    'Waiting': 0.2,         # 等待基本静态
    'WalkDog': 0.9,         # 行走需要节奏
    'Walking': 1.0,         # 步态节奏最重要
    'WalkTogether': 0.9,
}

# 维度权重（5个维度的相对重要性，总和=1.0）
DIMENSION_WEIGHTS = {
    'Directions': {'angle': 0.25, 'amplitude': 0.25, 'symmetry': 0.20, 'rhythm': 0.15, 'stability': 0.15},
    'Discussion': {'angle': 0.20, 'amplitude': 0.20, 'symmetry': 0.25, 'rhythm': 0.20, 'stability': 0.15},
    'Eating': {'angle': 0.30, 'amplitude': 0.25, 'symmetry': 0.10, 'rhythm': 0.20, 'stability': 0.15},
    'Greeting': {'angle': 0.20, 'amplitude': 0.30, 'symmetry': 0.20, 'rhythm': 0.20, 'stability': 0.10},
    'Phoning': {'angle': 0.35, 'amplitude': 0.20, 'symmetry': 0.05, 'rhythm': 0.15, 'stability': 0.25},
    'Photo': {'angle': 0.30, 'amplitude': 0.20, 'symmetry': 0.25, 'rhythm': 0.10, 'stability': 0.15},
    'Posing': {'angle': 0.25, 'amplitude': 0.15, 'symmetry': 0.35, 'rhythm': 0.05, 'stability': 0.20},
    'Purchases': {'angle': 0.25, 'amplitude': 0.25, 'symmetry': 0.20, 'rhythm': 0.15, 'stability': 0.15},
    'Sitting': {'angle': 0.35, 'amplitude': 0.10, 'symmetry': 0.30, 'rhythm': 0.05, 'stability': 0.20},
    'SittingDown': {'angle': 0.35, 'amplitude': 0.25, 'symmetry': 0.05, 'rhythm': 0.25, 'stability': 0.10},
    'Smoking': {'angle': 0.30, 'amplitude': 0.25, 'symmetry': 0.05, 'rhythm': 0.20, 'stability': 0.20},
    'Waiting': {'angle': 0.25, 'amplitude': 0.10, 'symmetry': 0.30, 'rhythm': 0.10, 'stability': 0.25},
    'WalkDog': {'angle': 0.25, 'amplitude': 0.25, 'symmetry': 0.20, 'rhythm': 0.20, 'stability': 0.10},
    'Walking': {'angle': 0.25, 'amplitude': 0.20, 'symmetry': 0.25, 'rhythm': 0.25, 'stability': 0.05},
    'WalkTogether': {'angle': 0.25, 'amplitude': 0.20, 'symmetry': 0.25, 'rhythm': 0.25, 'stability': 0.05},
}

# 稳定性阈值（重心移动范围，米）：超过阈值开始扣分
STABILITY_THRESHOLDS = {
    'Directions': 0.8,
    'Discussion': 0.8,
    'Eating': 0.5,
    'Greeting': 1.0,
    'Phoning': 0.6,
    'Photo': 0.5,
    'Posing': 0.3,          # 摆姿势应尽量静止
    'Purchases': 0.8,
    'Sitting': 0.4,         # 坐姿应稳定
    'SittingDown': 0.6,
    'Smoking': 0.5,
    'Waiting': 0.5,         # 等待应保持稳定
    'WalkDog': 1.5,         # 行走允许较大移动
    'Walking': 1.5,
    'WalkTogether': 1.5,
}


def get_action_config(action_name: str) -> Dict:
    """获取指定动作的配置"""
    return {
        'symmetry_weight': SYMMETRY_WEIGHTS.get(action_name, 0.5),
        'rhythm_weight': RHYTHM_WEIGHTS.get(action_name, 0.5),
        'dimension_weights': DIMENSION_WEIGHTS.get(action_name, {
            'angle': 0.30, 'amplitude': 0.25, 'symmetry': 0.20,
            'rhythm': 0.15, 'stability': 0.10
        }),
        'stability_threshold': STABILITY_THRESHOLDS.get(action_name, 1.0)
    }
