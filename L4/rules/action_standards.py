"""
动作标准定义 - 针对Human3.6M的15个动作类别定义评分规则
"""
from typing import Dict

H36M_ACTIONS = [
    'Directions', 'Discussion', 'Eating', 'Greeting',
    'Phoning', 'Photo', 'Posing', 'Purchases',
    'Sitting', 'SittingDown', 'Smoking', 'Waiting',
    'WalkDog', 'Walking', 'WalkTogether'
]

JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
    'Spine', 'Thorax', 'Neck/Nose', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]


class ActionStandard:
    """单个动作的标准定义"""
    def __init__(self, action_name: str):
        self.action_name = action_name
        self.angle_ranges = {}
        self.motion_ranges = {}
        self.symmetry_weight = 0.0
        self.rhythm_weight = 0.0
    
    def set_angle_range(self, joint: str, min_angle: float, max_angle: float):
        self.angle_ranges[joint] = (min_angle, max_angle)
    
    def set_motion_range(self, joint_name: str, min_range: float):
        self.motion_ranges[joint_name] = min_range
    
    def set_symmetry_weight(self, weight: float):
        self.symmetry_weight = weight
    
    def set_rhythm_weight(self, weight: float):
        self.rhythm_weight = weight


def create_action_standards() -> Dict[str, ActionStandard]:
    """为15个动作创建标准规则"""
    standards = {}
    
    # Directions (指路)
    directions = ActionStandard('Directions')
    directions.set_angle_range('right_elbow', 30, 150)
    directions.set_angle_range('left_elbow', 30, 150)
    directions.set_angle_range('right_knee', 150, 180)
    directions.set_angle_range('left_knee', 150, 180)
    directions.set_motion_range('RWrist', 0.3)
    directions.set_symmetry_weight(0.6)
    directions.set_rhythm_weight(0.3)
    standards['Directions'] = directions
    
    # Discussion (讨论)
    discussion = ActionStandard('Discussion')
    discussion.set_angle_range('right_knee', 150, 180)
    discussion.set_angle_range('left_knee', 150, 180)
    discussion.set_motion_range('Head', 0.1)
    discussion.set_symmetry_weight(0.5)
    discussion.set_rhythm_weight(0.4)
    standards['Discussion'] = discussion
    
    # 其他13个动作（简化版）
    for action in ['Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 
                   'Smoking', 'Waiting']:
        std = ActionStandard(action)
        std.set_angle_range('right_knee', 150, 180)
        std.set_angle_range('left_knee', 150, 180)
        std.set_symmetry_weight(0.5)
        std.set_rhythm_weight(0.3)
        standards[action] = std
    
    # Sitting (坐着)
    sitting = ActionStandard('Sitting')
    sitting.set_angle_range('right_knee', 60, 110)
    sitting.set_angle_range('left_knee', 60, 110)
    sitting.set_angle_range('right_hip', 70, 110)
    sitting.set_angle_range('left_hip', 70, 110)
    sitting.set_symmetry_weight(0.9)
    sitting.set_rhythm_weight(0.1)
    standards['Sitting'] = sitting
    
    # SittingDown (坐下)
    sitting_down = ActionStandard('SittingDown')
    sitting_down.set_angle_range('right_knee', 60, 170)
    sitting_down.set_angle_range('left_knee', 60, 170)
    sitting_down.set_motion_range('Hip', 0.4)
    sitting_down.set_symmetry_weight(0.8)
    sitting_down.set_rhythm_weight(0.7)
    standards['SittingDown'] = sitting_down
    
    # Walking类动作
    for action in ['WalkDog', 'Walking', 'WalkTogether']:
        walk = ActionStandard(action)
        walk.set_angle_range('right_knee', 130, 180)
        walk.set_angle_range('left_knee', 130, 180)
        walk.set_motion_range('RFoot', 0.5)
        walk.set_motion_range('LFoot', 0.5)
        walk.set_symmetry_weight(0.8)
        walk.set_rhythm_weight(0.9)
        standards[action] = walk
    
    return standards


ACTION_STANDARDS = create_action_standards()


def get_standard(action_name: str) -> ActionStandard:
    """获取指定动作的标准"""
    return ACTION_STANDARDS.get(action_name, ActionStandard(action_name))