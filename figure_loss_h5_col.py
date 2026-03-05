import h5py
import numpy as np
import gym, yumi_gym
# from yumi_gym import YumiEnv
import pybullet as p
import time

def calculate_collision_ratio_from_h5(h5_file_path, joints_key='joint_angles', max_steps=None):
    """
    从H5文件中读取关节角度数据，逐帧计算碰撞并返回总碰撞比例
    
    Args:
        h5_file_path: H5文件路径
        joints_key: H5文件中存储关节角度的键名，默认为'joint_angles'
        max_steps: 最大处理步数，如果为None则处理全部数据
    
    Returns:
        tuple: (collision_ratio, total_frames, collision_frames)
    """
    # 创建环境实例
    env = gym.make('yumi-v0')
    
    # 重置环境
    env.reset()
    
    # 读取H5文件中的关节角度数据
    with h5py.File(h5_file_path, 'r') as f:
        joint_angles_data = f[joints_key][:]  # 假设数据形状为 [frames, num_joints]
    
    #补五列零对齐仿真环境
    print(joint_angles_data.shape)
    if joint_angles_data.shape[1] < 23:
        padding = np.zeros((joint_angles_data.shape[0], 23 - joint_angles_data.shape[1]))
        joint_angles_data = np.hstack((joint_angles_data, padding))

    # 如果指定了最大步数，则限制处理范围
    if max_steps is not None:
        joint_angles_data = joint_angles_data[:max_steps]
    
    total_frames = len(joint_angles_data)
    collision_count = 0
    
    print(f"开始处理 {total_frames} 帧数据...")
    
    for frame_idx, action in enumerate(joint_angles_data):
        # 在每次执行动作前重置环境
        env.render()
        # 执行动作（设置关节角度）
        obs, reward, done, info = env.step(action)
        
        # 检查是否发生碰撞
        if info['collision']:
            collision_count += 1
        
        # 每100帧输出一次进度
        if (frame_idx + 1) % 100 == 0:
            print(f"已处理 {frame_idx + 1}/{total_frames} 帧，碰撞帧数: {collision_count}")
        print(f"当前帧: {frame_idx}")
        time.sleep(0.001)
    # 计算碰撞比例
    collision_ratio = collision_count / total_frames if total_frames > 0 else 0
    
    print(f"\n处理完成！")
    print(f"总帧数: {total_frames}")
    print(f"碰撞帧数: {collision_count}")
    print(f"碰撞比例: {collision_ratio:.4f} ({collision_ratio * 100:.2f}%)")
    
    # 关闭环境
    env.close()
    
    return collision_ratio, total_frames, collision_count


def calculate_collision_with_custom_reward(h5_file_path, joints_key='joint_angles', max_steps=None):
    """
    使用自定义奖励函数的方式计算碰撞
    
    Args:
        h5_file_path: H5文件路径
        joints_key: H5文件中存储关节角度的键名
        max_steps: 最大处理步数
    
    Returns:
        dict: 包含碰撞统计信息的字典
    """
    # 创建环境实例
    env = gym.make('yumi-v0')
    
    # 重置环境
    env.reset()
    
    # 读取H5文件中的关节角度数据
    with h5py.File(h5_file_path, 'r') as f:
        joint_angles_data = f[joints_key][:]
    
    # 如果指定了最大步数，则限制处理范围
    if max_steps is not None:
        joint_angles_data = joint_angles_data[:max_steps]
    
    total_frames = len(joint_angles_data)
    collision_count = 0
    collision_details = []  # 存储碰撞详细信息
    
    print(f"开始处理 {total_frames} 帧数据...")
    
    for frame_idx, action in enumerate(joint_angles_data):
        # 定义自定义奖励函数来检测碰撞
        def custom_collision_check(jointStates, collision, step_counter):
            done = False
            reward = -1 if collision else 0  # 发生碰撞给予负奖励
            return reward, done
        
        # 执行动作并使用自定义奖励函数
        obs, reward, done, info = env.step(action, custom_reward=custom_collision_check)
        
        # 检查是否发生碰撞
        if info['collision']:
            collision_count += 1
            collision_details.append({
                'frame': frame_idx,
                'action': action.copy(),
                'reward': reward
            })
        
        # 每100帧输出一次进度
        if (frame_idx + 1) % 100 == 0:
            print(f"已处理 {frame_idx + 1}/{total_frames} 帧，碰撞帧数: {collision_count}")
    
    # 计算碰撞比例
    collision_ratio = collision_count / total_frames if total_frames > 0 else 0
    
    result = {
        'collision_ratio': collision_ratio,
        'total_frames': total_frames,
        'collision_frames': collision_count,
        'collision_details': collision_details
    }
    
    print(f"\n处理完成！")
    print(f"总帧数: {total_frames}")
    print(f"碰撞帧数: {collision_count}")
    print(f"碰撞比例: {collision_ratio:.4f} ({collision_ratio * 100:.2f}%)")
    
    # 关闭环境
    env.close()
    
    return result


# def enable_collision_detection_in_env(env):
#     """
#     启用环境中的碰撞检测功能
#     """
#     # 这个函数可以用来修改环境使其启用碰撞检测
#     # 注意：原始代码中碰撞检测部分是被注释掉的
#     pass


if __name__ == "__main__":
    # 使用示例
    from config.variables_define import *
    # ab_experiment_name = 'None'
    # ab_experiment_name = 'ab_vec_loss'
    # ab_experiment_name = 'ab_tip_pos_loss'
    ab_experiment_name = 'ab_col_loss'
    # ab_experiment_name = 'ab_tip_dis_loss'
    h5_file_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_output.h5"
    # h5_file_path = "your_motion_data.h5"  # 替换为您的H5文件路径
    joints_key = "outputs"  # 替换为H5文件中正确的键名
    
    try:
        # 方法1：基本碰撞检测
        collision_ratio, total_frames, collision_count = calculate_collision_ratio_from_h5(
            h5_file_path=h5_file_path,
            joints_key=joints_key
        )
        
        # 方法2：使用自定义奖励函数（提供更多细节）
        # result = calculate_collision_with_custom_reward(
        #     h5_file_path=h5_file_path,
        #     joints_key=joints_key
        # )
        # print(f"碰撞详情: {result}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# 示例用法
# if __name__ == "__main__":
#     # 替换为实际的文件路径
#     file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_visionpro.h5"
#     # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output1.h5"
#     file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5"
#     # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output1.h5"
#     # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output2.h5"
#     # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_slahmr.h5"
#     # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output.h5"
#     # # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output1.h5"
#     # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output2.h5"
#     # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_visionpro10000.h5"
#     # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_outputVP2.h5"
#     try:
#         mse_per_joint = compare_outputs_mse_per_joint(file1_path, file2_path)
#         print(f"\n所有关节的均方差数组: {mse_per_joint}")
#     except ValueError as e:
#         print(f"错误: {e}")
#     except FileNotFoundError as e:
#         print(f"文件未找到: {e}")
#     except Exception as e:
#         print(f"处理过程中发生错误: {e}")