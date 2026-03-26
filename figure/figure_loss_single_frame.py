import gym, yumi_gym
import pybullet as p
import pybullet_data
from config.variables_define import *

import h5py
import numpy as np

# 从h5文件加载第一帧的指尖数据
def load_human_tip_from_h5(h5_file_path, subject_key="测试-ceshi", dataset_name='r_glove_pos'):
    with h5py.File(h5_file_path, 'r') as f:
        if subject_key not in f:
            raise KeyError(f"Subject key '{subject_key}' not found in file")
        
        subject_data = f[subject_key]
        
        if dataset_name not in subject_data:
            raise KeyError(f"Dataset '{dataset_name}' not found in subject data")
        
        data = subject_data[dataset_name]
        
        # 提取第一帧数据
        if len(data.shape) == 3:  # [帧数, 关节数, 3]
            first_frame = data[0]  # [关节数, 3]
        elif len(data.shape) == 2:  # [关节数, 3]
            first_frame = data
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    
    return first_frame

env = gym.make('yumi-v0')
observation = env.reset()
# 加载h5文件数据
h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204_aligned.h5"  # 根据实际路径修改
first_frame_data = load_human_tip_from_h5(h5_file_path, subject_key="测试-ceshi")

# 根据TIP_dic获取指尖位置
human_tip_pos = [first_frame_data[idx].tolist() for idx in TIP_dic]

print("从h5文件第一帧提取的指尖位置:", human_tip_pos)

robot_path = urdf_file
# 连接 PyBullet（GUI 模式方便调试，DIRECT 模式适合大规模数据预处理）
robot_id = env.yumiUid

# 获取关节数量和限位（这一步是为了让 IK 结果符合物理规律）
num_joints = p.getNumJoints(robot_id)
print("机器人关节数量:", num_joints)
# 打印所有关节信息以便调试
print("\n所有关节信息:")
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    joint_type = info[2]
    lower_limit = info[8]
    upper_limit = info[9]
    
    # 输出关节类型名称
    joint_type_names = {
        p.JOINT_REVOLUTE: 'REVOLUTE',
        p.JOINT_PRISMATIC: 'PRISMATIC', 
        p.JOINT_SPHERICAL: 'SPHERICAL',
        p.JOINT_PLANAR: 'PLANAR',
        p.JOINT_FIXED: 'FIXED',
        p.JOINT_POINT2POINT: 'POINT2POINT',
        p.JOINT_GEAR: 'GEAR'
    }
    
    joint_type_name = joint_type_names.get(joint_type, 'UNKNOWN')
    
    print(f"关节 {i}: 名称={joint_name}, 类型={joint_type_name}, 下限={lower_limit}, 上限={upper_limit}")

lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    # 获取关节类型，只有可移动关节才有位置限制
    joint_type = info[2]
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:  # 只处理旋转关节和平移关节
        lower_limit = info[8]
        upper_limit = info[9]
        # 确保下限小于上限
        if lower_limit > upper_limit:
            lower_limit, upper_limit = upper_limit, lower_limit
        
        lower_limits.append(lower_limit)
        upper_limits.append(upper_limit)
        joint_ranges.append(upper_limit - lower_limit)
        rest_poses.append((upper_limit + lower_limit) / 2)  # 初始位置设为中值
    else:
        # 对于固定关节、连续关节等，设置默认值
        lower_limits.append(0.0)  # 对于连续关节或无限制关节使用0
        upper_limits.append(0.0)
        joint_ranges.append(0.0)
        rest_poses.append(0.0)  # 对于连续关节设置为0
        
print("下限:", lower_limits)
print('1')
print("上限:", upper_limits)

# 映射逻辑：将人手坐标变换到机器人基座坐标系下
# 这里通常需要加一个 offset 或 scale
target_pos = []
for h in human_tip_pos:
    scaled_pos = [coord * scaling_factor / scaling_factor_rb for coord in h]  # 对每个坐标分量进行缩放
    target_pos.append(scaled_pos)

# 或者使用numpy向量化操作
# target_pos = (np.array(human_tip_pos) * 1.2).tolist()

# 假设食指尖的 link index 是 5
finger_tip_indices = TIP_dic_rb_gym  # 根据实际机器人模型调整索引
# target_positions = [[...], [...], [...], [...], [...]] # 五根手指的目标位置
target_positions = target_pos
# 计算 IK
# 注意：PyBullet 的 IK 默认返回所有可移动关节的角度
full_joint_angles = p.calculateInverseKinematics2(
    robot_id, 
    finger_tip_indices, 
    target_positions,
    lowerLimits=lower_limits, 
    upperLimits=upper_limits
)
print("计算得到的关节角度:", full_joint_angles)