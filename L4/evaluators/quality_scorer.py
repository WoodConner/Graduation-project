"""
动作质量评分器
综合5个维度对动作执行质量进行评分
"""
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from metrics.joint_angles import (
    compute_all_angles, compute_symmetry_score,
    compute_body_tilt, compute_motion_range,
    compute_velocity_smoothness, compute_support_stability
)
from rules.action_standards import get_standard, JOINT_NAMES


class QualityScorer:
    """动作质量评分器"""
    
    def __init__(self, action_name: str):
        self.action_name = action_name
        self.standard = get_standard(action_name)
        
        # 5个维度的权重
        self.weights = {
            'angle': 0.30,      # 关节角度规范性
            'amplitude': 0.25,  # 动作幅度
            'symmetry': 0.20,   # 左右对称性
            'rhythm': 0.15,     # 节奏与平滑性
            'stability': 0.10   # 姿态稳定性
        }
        
    def score_angles(self, sequence: np.ndarray) -> float:
        """评估关节角度规范性 (0-30)"""
        if len(self.standard.angle_ranges) == 0:
            return 30.0
        
        scores = []
        for t in range(len(sequence)):
            angles = compute_all_angles(sequence[t])
            frame_scores = []
            for joint, (min_angle, max_angle) in self.standard.angle_ranges.items():
                if joint not in angles:
                    continue
                actual_angle = angles[joint]
                if min_angle <= actual_angle <= max_angle:
                    frame_scores.append(1.0)
                else:
                    deviation = min(abs(actual_angle - min_angle), abs(actual_angle - max_angle))
                    score = max(0.0, 1.0 - deviation / 30.0)
                    frame_scores.append(score)
            if frame_scores:
                scores.append(np.mean(frame_scores))
        return np.mean(scores) * 30.0 if scores else 30.0
    
    def score_amplitude(self, sequence: np.ndarray) -> float:
        """评估动作幅度 (0-25)"""
        if len(self.standard.motion_ranges) == 0:
            return 25.0
        scores = []
        for joint_name, min_range in self.standard.motion_ranges.items():
            if joint_name not in JOINT_NAMES:
                continue
            joint_idx = JOINT_NAMES.index(joint_name)
            actual_range = compute_motion_range(sequence, joint_idx)
            if actual_range >= min_range:
                scores.append(1.0)
            else:
                scores.append(actual_range / (min_range + 1e-8))
        return np.mean(scores) * 25.0 if scores else 25.0
    
    def score_symmetry(self, sequence: np.ndarray) -> float:
        """评估左右对称性 (0-20)"""
        symmetry_scores = []
        for t in range(len(sequence)):
            angles = compute_all_angles(sequence[t])
            sym = compute_symmetry_score(angles)
            symmetry_scores.append(np.mean(list(sym.values())))
        avg_symmetry = np.mean(symmetry_scores)
        weighted_score = avg_symmetry * self.standard.symmetry_weight + \
                        (1.0 - self.standard.symmetry_weight) * 0.5
        return weighted_score * 20.0
    
    def score_rhythm(self, sequence: np.ndarray) -> float:
        """评估节奏与平滑性 (0-15)"""
        key_joints = [2, 5, 12, 15]
        smoothness_scores = []
        for joint_idx in key_joints:
            smoothness = compute_velocity_smoothness(sequence, joint_idx)
            score = max(0.0, 1.0 - smoothness / 0.01)
            smoothness_scores.append(score)
        avg_smoothness = np.mean(smoothness_scores)
        weighted_score = avg_smoothness * self.standard.rhythm_weight + \
                        (1.0 - self.standard.rhythm_weight) * 0.5
        return weighted_score * 15.0
    
    def score_stability(self, sequence: np.ndarray) -> float:
        """评估姿态稳定性 (0-10)"""
        com_range = compute_support_stability(sequence)
        tilts = [compute_body_tilt(sequence[t]) for t in range(len(sequence))]
        avg_shoulder_tilt = np.mean([t[0] for t in tilts])
        avg_hip_tilt = np.mean([t[1] for t in tilts])
        com_score = max(0.0, 1.0 - com_range / 3.0)
        shoulder_score = max(0.0, 1.0 - avg_shoulder_tilt / 0.2)
        hip_score = max(0.0, 1.0 - avg_hip_tilt / 0.15)
        stability_score = (com_score * 0.5 + shoulder_score * 0.3 + hip_score * 0.2)
        return stability_score * 10.0
    
    def score(self, sequence: np.ndarray, verbose: bool = False) -> dict:
        """综合评分"""
        angle_score = self.score_angles(sequence)
        amplitude_score = self.score_amplitude(sequence)
        symmetry_score = self.score_symmetry(sequence)
        rhythm_score = self.score_rhythm(sequence)
        stability_score = self.score_stability(sequence)
        total_score = angle_score + amplitude_score + symmetry_score + rhythm_score + stability_score
        result = {
            'total': total_score,
            'angle': angle_score,
            'amplitude': amplitude_score,
            'symmetry': symmetry_score,
            'rhythm': rhythm_score,
            'stability': stability_score,
            'action': self.action_name
        }
        if verbose:
            print(f"\n{'='*50}")
            print(f"动作: {self.action_name}")
            print(f"{'='*50}")
            print(f"关节角度规范性: {angle_score:.1f}/30")
            print(f"动作幅度:        {amplitude_score:.1f}/25")
            print(f"左右对称性:      {symmetry_score:.1f}/20")
            print(f"节奏与平滑性:    {rhythm_score:.1f}/15")
            print(f"姿态稳定性:      {stability_score:.1f}/10")
            print(f"{'-'*50}")
            print(f"总分:            {total_score:.1f}/100")
            print(f"{'='*50}\n")
        return result


def batch_score(sequences: list, labels: list, action_names: list, verbose: bool = False) -> list:
    """批量评分"""
    results = []
    for seq, label in zip(sequences, labels):
        action_name = action_names[label]
        scorer = QualityScorer(action_name)
        result = scorer.score(seq, verbose=verbose)
        results.append(result)
    return results
