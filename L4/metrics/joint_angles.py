"""
关节角度计算工具
支持常见关节角度计算，用于动作规范性评估
"""
import numpy as np
from typing import Dict, Tuple


# Human3.6M 17关节索引定义 (从GT 32映射后)
JOINT_NAMES = [
    'Hip',           # 0
    'RHip',          # 1
    'RKnee',         # 2
    'RFoot',         # 3
    'LHip',          # 4
    'LKnee',         # 5
    'LFoot',         # 6
    'Spine',         # 7
    'Thorax',        # 8
    'Neck/Nose',     # 9
    'Head',          # 10
    'LShoulder',     # 11
    'LElbow',        # 12
    'LWrist',        # 13
    'RShoulder',     # 14
    'RElbow',        # 15
    'RWrist',        # 16
]


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个3D向量的夹角（度数）
    
    Args:
        v1, v2: shape (3,) 的向量
    
    Returns:
        角度值（0-180度）
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def compute_joint_angle(pose: np.ndarray, joint_a: int, joint_b: int, joint_c: int) -> float:
    """
    计算三点形成的关节角度
    
    Args:
        pose: shape (17, 3) 的3D姿态
        joint_a, joint_b, joint_c: 关节索引，计算b点的角度（a-b-c）
    
    Returns:
        关节角度（度数）
    """
    vec_ba = pose[joint_a] - pose[joint_b]  # b指向a的向量
    vec_bc = pose[joint_c] - pose[joint_b]  # b指向c的向量
    return vector_angle(vec_ba, vec_bc)


def compute_all_angles(pose: np.ndarray) -> Dict[str, float]:
    """
    计算所有常用关节角度
    
    Args:
        pose: shape (17, 3) 的3D姿态
    
    Returns:
        角度字典
    """
    angles = {}
    
    # 膝关节角度
    angles['right_knee'] = compute_joint_angle(pose, 1, 2, 3)    # RHip-RKnee-RFoot
    angles['left_knee'] = compute_joint_angle(pose, 4, 5, 6)     # LHip-LKnee-LFoot
    
    # 髋关节角度（大腿与躯干）
    angles['right_hip'] = compute_joint_angle(pose, 7, 1, 2)     # Spine-RHip-RKnee
    angles['left_hip'] = compute_joint_angle(pose, 7, 4, 5)      # Spine-LHip-LKnee
    
    # 肘关节角度
    angles['right_elbow'] = compute_joint_angle(pose, 14, 15, 16) # RShoulder-RElbow-RWrist
    angles['left_elbow'] = compute_joint_angle(pose, 11, 12, 13)  # LShoulder-LElbow-LWrist
    
    # 肩关节角度（上臂与躯干）
    angles['right_shoulder'] = compute_joint_angle(pose, 8, 14, 15) # Thorax-RShoulder-RElbow
    angles['left_shoulder'] = compute_joint_angle(pose, 8, 11, 12)  # Thorax-LShoulder-LElbow
    
    # 脊柱角度（躯干弯曲）
    angles['spine'] = compute_joint_angle(pose, 0, 7, 8)         # Hip-Spine-Thorax
    
    # 颈部角度
    angles['neck'] = compute_joint_angle(pose, 7, 8, 9)          # Spine-Thorax-Neck
    
    return angles


def compute_symmetry_score(angles: Dict[str, float]) -> Dict[str, float]:
    """
    计算左右对称性分数
    
    Args:
        angles: 关节角度字典
    
    Returns:
        对称性分数字典（0-1，越接近1越对称）
    """
    symmetry = {}
    
    pairs = [
        ('knee', 'left_knee', 'right_knee'),
        ('hip', 'left_hip', 'right_hip'),
        ('elbow', 'left_elbow', 'right_elbow'),
        ('shoulder', 'left_shoulder', 'right_shoulder'),
    ]
    
    for name, left_key, right_key in pairs:
        left_angle = angles.get(left_key, 0)
        right_angle = angles.get(right_key, 0)
        diff = abs(left_angle - right_angle)
        # 差异小于5度得满分，差异30度以上得0分
        score = max(0.0, 1.0 - diff / 30.0)
        symmetry[name] = score
    
    return symmetry


def compute_body_tilt(pose: np.ndarray) -> Tuple[float, float]:
    """
    计算身体倾斜角度
    
    Args:
        pose: shape (17, 3)
    
    Returns:
        (肩部水平差, 髋部水平差) 单位：米
    """
    # 肩部水平差（左右肩Y坐标差）
    shoulder_tilt = abs(pose[11, 1] - pose[14, 1])  # LShoulder vs RShoulder
    
    # 髋部水平差
    hip_tilt = abs(pose[4, 1] - pose[1, 1])  # LHip vs RHip
    
    return shoulder_tilt, hip_tilt


def compute_motion_range(sequence: np.ndarray, joint_idx: int) -> float:
    """
    计算某关节在序列中的运动范围
    
    Args:
        sequence: shape (T, 17, 3)
        joint_idx: 关节索引
    
    Returns:
        运动范围（米）
    """
    positions = sequence[:, joint_idx, :]  # (T, 3)
    range_x = positions[:, 0].max() - positions[:, 0].min()
    range_y = positions[:, 1].max() - positions[:, 1].min()
    range_z = positions[:, 2].max() - positions[:, 2].min()
    return np.sqrt(range_x**2 + range_y**2 + range_z**2)


def compute_velocity_smoothness(sequence: np.ndarray, joint_idx: int) -> float:
    """
    计算关节运动速度的平滑性（越小越平滑）
    
    Args:
        sequence: shape (T, 17, 3)
        joint_idx: 关节索引
    
    Returns:
        加速度方差（平滑性指标）
    """
    positions = sequence[:, joint_idx, :]  # (T, 3)
    
    # 一阶差分：速度
    velocities = np.diff(positions, axis=0)
    
    # 二阶差分：加速度
    accelerations = np.diff(velocities, axis=0)
    
    # 加速度幅值的方差（越小越平滑）
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    smoothness = np.var(acc_magnitudes)
    
    return smoothness


def compute_center_of_mass(pose: np.ndarray) -> np.ndarray:
    """
    计算身体重心（简化为所有关节的平均位置）
    
    Args:
        pose: shape (17, 3)
    
    Returns:
        重心坐标 (3,)
    """
    return pose.mean(axis=0)


def compute_support_stability(sequence: np.ndarray) -> float:
    """
    计算重心稳定性（重心在序列中的移动范围）
    
    Args:
        sequence: shape (T, 17, 3)
    
    Returns:
        重心移动范围（米）
    """
    com_trajectory = np.array([compute_center_of_mass(pose) for pose in sequence])
    com_range = np.linalg.norm(com_trajectory.max(axis=0) - com_trajectory.min(axis=0))
    return com_range
