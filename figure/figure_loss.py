import gym, yumi_gym
import pybullet as p
import pybullet_data
from config.variables_define import *

import h5py
import numpy as np

#     所有关节信息:
# 关节 0: 名称=index_mcp_roll, 类型=REVOLUTE, 下限=-0.18, 上限=0.18
# 关节 1: 名称=index_mcp_pitch, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 2: 名称=index_pip, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 3: 名称=middle_mcp_roll, 类型=REVOLUTE, 下限=-0.18, 上限=0.18
# 关节 4: 名称=middle_mcp_pitch, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 5: 名称=middle_pip, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 6: 名称=ring_mcp_roll, 类型=REVOLUTE, 下限=-0.18, 上限=0.18
# 关节 7: 名称=ring_mcp_pitch, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 8: 名称=ring_pip, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 9: 名称=pinky_mcp_roll, 类型=REVOLUTE, 下限=-0.18, 上限=0.18
# 关节 10: 名称=pinky_mcp_pitch, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 11: 名称=pinky_pip, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 12: 名称=thumb_cmc_roll, 类型=REVOLUTE, 下限=-0.6, 上限=0.6
# 关节 13: 名称=thumb_cmc_yaw, 类型=REVOLUTE, 下限=0.0, 上限=1.6
# 关节 14: 名称=thumb_cmc_pitch, 类型=REVOLUTE, 下限=0.0, 上限=1.0
# 关节 15: 名称=thumb_mcp, 类型=REVOLUTE, 下限=0.0, 上限=1.57
# 关节 16: 名称=thumb_ip, 类型=REVOLUTE, 下限=0.0, 上限=1.57
dict_adjust = {
        # 食指关节 (index finger)
        0: 1,  # index_mcp_roll (PyBullet关节0 -> Linker Hand关节1)
        1: 2,  # index_mcp_pitch (PyBullet关节1 -> Linker Hand关节2)
        2: 3,  # index_pip (PyBullet关节2 -> Linker Hand关节3)
        
        # 中指关节 (middle finger)
        3: 4,  # middle_mcp_roll (PyBullet关节3 -> Linker Hand关节4)
        4: 5,  # middle_mcp_pitch (PyBullet关节4 -> Linker Hand关节5)
        5: 6,  # middle_pip (PyBullet关节5 -> Linker Hand关节6)
        
        # 无名指关节 (ring finger)
        6: 7,  # ring_mcp_roll (PyBullet关节6 -> Linker Hand关节7)
        7: 8,  # ring_mcp_pitch (PyBullet关节7 -> Linker Hand关节8)
        8: 9,  # ring_pip (PyBullet关节8 -> Linker Hand关节9)
        
        # 小指关节 (pinky finger)
        9: 10, # pinky_mcp_roll (PyBullet关节9 -> Linker Hand关节10)
        10: 11, # pinky_mcp_pitch (PyBullet关节10 -> Linker Hand关节11)
        11: 12, # pinky_pip (PyBullet关节11 -> Linker Hand关节12)
        
        # 拇指关节 (thumb)
        12: 13, # thumb_cmc_roll (PyBullet关节12 -> Linker Hand关节13)
        13: 14, # thumb_cmc_yaw (PyBullet关节13 -> Linker Hand关节14)
        14: 15, # thumb_cmc_pitch (PyBullet关节14 -> Linker Hand关节15)
        15: 16, # thumb_mcp (PyBullet关节15 -> Linker Hand关节16)
        16: 17 # thumb_ip (PyBullet关节16 -> Linker Hand关节17)
        
        # # 指尖关节 (tips) - 如果需要的话
        # 17: 18, # index_tip
        # 18: 19, # middle_tip
        # 19: 20, # ring_tip
        # 20: 21, # pinky_tip
        # 21: 22  # thumb_tip
    }
# 从h5文件加载所有帧的指尖数据
def load_human_tip_from_h5_all_frames(h5_file_path, subject_key="测试-ceshi", dataset_name='r_glove_pos'):
    with h5py.File(h5_file_path, 'r') as f:
        if subject_key not in f:
            raise KeyError(f"Subject key '{subject_key}' not found in file")
        
        subject_data = f[subject_key]
        
        if dataset_name not in subject_data:
            raise KeyError(f"Dataset '{dataset_name}' not found in subject data")
        
        data = subject_data[dataset_name]
        
        # 数据应为 [帧数, 关节数, 3]
        if len(data.shape) == 3:  # [帧数, 关节数, 3]
            all_frames = data[:]  # 使用 [:] 读取整个数据集为 NumPy 数组
        elif len(data.shape) == 2:  # [关节数, 3] - 单帧数据
            all_frames = data[np.newaxis, :, :]  # 添加帧维度
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
    
    return all_frames  # 现在返回的是 NumPy 数组，不是 HDF5 数据集对象
def adjust_angles(joint_angles):
    # 新建一个列表全为0 长度为out_num_joint
    adjusted_angles = np.zeros(out_num_joint)
    
    # 应用映射关系并限制关节角度
    for i in range(len(joint_angles)):
        if i in dict_adjust:
            joint_idx = dict_adjust[i]
            angle = joint_angles[i]
            
            # 使用 angle_limit_rob 限位 (从 variables_define.py 导入)
            if joint_idx < len(angle_limit_rob):
                lower_bound = angle_limit_rob[joint_idx][0]
                upper_bound = angle_limit_rob[joint_idx][1]
                angle = np.clip(angle, lower_bound, upper_bound)
            
            adjusted_angles[joint_idx] = angle
    
    return adjusted_angles

# 初始化环境
env = gym.make('yumi-v0')
observation = env.reset()

# 获取机器人ID
robot_id = env.yumiUid

# 获取关节数量和限位（这一步是为了让 IK 结果符合物理规律）
num_joints = p.getNumJoints(robot_id)
print("机器人关节数量:", num_joints)

# 创建关节名称到索引的映射
joint_name_to_index = {}
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    joint_name_to_index[joint_name] = i

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
            print(f"警告: 关节 {info[1].decode('utf-8')} ({i}) 的下限大于上限，已自动修正")
            lower_limit, upper_limit = upper_limit, lower_limit
        
        lower_limits.append(lower_limit)
        upper_limits.append(upper_limit)
        joint_ranges.append(upper_limit - lower_limit)
        rest_poses.append((upper_limit + lower_limit) / 2)  # 初始位置设为中值
    else:
        # 对于固定关节、连续关节等，设置默认值
        lower_limits.append(-1e10)  # 使用大范围而不是0
        upper_limits.append(1e10)
        joint_ranges.append(2e10)
        rest_poses.append(0.0)

print("下限:", lower_limits)
print('1')
print("上限:", upper_limits)

# 验证指尖关节索引是否正确
print("\n指尖关节在PyBullet中的索引:")
tip_joint_names = ['index_tip', 'middle_tip', 'ring_tip', 'pinky_tip', 'thumb_tip'] #linker
# tip_joint_names = ['thtip_joint', 'fftip_joint', 'mftip_joint', 'rftip_joint', 'lftip_joint' ] #svhhand
# tip_joint_names = ['THtip', 'FFtip', 'MFtip', 'RFtip', 'LFtip'] #shadow
for tip_name in tip_joint_names:
    if tip_name in joint_name_to_index:
        print(f"{tip_name} -> 索引 {joint_name_to_index[tip_name]}")
    else:
        print(f"{tip_name} -> 未找到")

# 使用正确的关节索引
finger_tip_indices = [joint_name_to_index[name] for name in tip_joint_names if name in joint_name_to_index]
print(f"\n用于IK计算的指尖关节索引: {finger_tip_indices}")

# # 加载h5文件数据（所有帧） 
# h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\glove_data_aligned.h5"
h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\test_glove_data0204_aligned.h5"
all_frames_data = load_human_tip_from_h5_all_frames(h5_file_path, subject_key="测试-ceshi")
# h5_file_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove_aligned.h5"
# all_frames_data = load_human_tip_from_h5_all_frames(h5_file_path, subject_key="S000079_P0008")
print(f"总共有 {len(all_frames_data)} 帧数据")

# 存储所有帧的输出角度
outputs = []
human_tips = source_dic['TIP_dic']
# human_tips = source_dic['PIP_dic']
# human_tips = np.array(human_tips)
# 遍历每一帧
for frame_idx, frame_data in enumerate(all_frames_data):
    # print(f"处理第 {frame_idx + 1}/{len(all_frames_data)} 帧...")
    
    # 根据TIP_dic获取当前帧的指尖位置
    human_tip_pos = [frame_data[idx].tolist() for idx in human_tips ]
    
    # 映射逻辑：将人手坐标变换到机器人基座坐标系下
    target_pos = []
    for h in human_tip_pos:
        # scaled_pos = [coord * scaling_factor / scaling_factor_rb for coord in h]  # 对每个坐标分量进行缩放
        scaled_pos = [coord * 1.3 for coord in h]  # 对每个坐标分量进行缩放
        target_pos.append(scaled_pos)
    
    # 计算 IK
    try:
        full_joint_angles = p.calculateInverseKinematics2(
            robot_id, 
            finger_tip_indices, 
            target_pos,
            lowerLimits=lower_limits, 
            upperLimits=upper_limits,
            maxNumIterations=400,      # 增加最大迭代次数
            residualThreshold=1e-7     # 提高精度阈值
        )
        #角度需要进行映射和调整，确保符合机器手的实际角度
        full_joint_angles = adjust_angles(full_joint_angles)
        outputs.append(full_joint_angles)
        print(f"  第{frame_idx+1}帧: 成功计算出{len(full_joint_angles)}个关节角度")
    except Exception as e:
        # print(f"  第{frame_idx+1}帧: IK计算失败 - {e}")
        # 如果IK失败，可以添加零值或上次成功的值
        if len(outputs) > 0:
            outputs.append(outputs[-1])  # 使用上一帧的结果
        else:
            outputs.append([0.0] * len(lower_limits))  # 使用零值

# 转换为numpy数组
outputs = np.array(outputs)
print(f"输出形状: {outputs.shape}")

# 保存到新的h5文件（只保存角度数据）
output_h5_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_{data_tpye}.h5"
# output_h5_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_{data_tpye}10000_t.h5"
with h5py.File(output_h5_path, 'w') as f:
    f.create_dataset('outputs', data=outputs)

print(f"输出数据已保存到: {output_h5_path}")
print(f"输出数据形状: {outputs.shape}")
print(f"前几个角度值示例: {outputs[0] if len(outputs) > 0 else 'No data'}")

# 关闭环境
env.close()
