import h5py
import numpy as np
from config.variables_define import *
from model.loss import RealTimeVisualizer
import matplotlib.pyplot as plt

scaling = 4.5/4.0  #缩放因子
def compare_outputs_distance_mean(file1_path, file2_path, dataset_name='r', dataset_name2='outputs', visualizer=None):
    """
    比较两个HDF5文件中outputs数据集的角度数组，计算两点间距离并求平均值
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径
        dataset_name: 数据集名称，默认为'r'
        dataset_name2: 第二个数据集名称，默认为'outputs'
        visualizer: 实时可视化器实例
    Returns:
        mean_distances_per_joint: 每个关节的距离平均值数组
    """
    # 打开两个HDF5文件
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # 获取outputs数据
        dataset = f1['测试-ceshi']
        outputs1 = dataset[dataset_name][:]
        outputs2 = f2[dataset_name2][:]
        
        outputs1 = outputs1 * scaling
        
        if visualizer is not None:
            # 只显示批次中的第一个样本
            actual_coords = outputs1[0]  # 实际手部坐标
            robot_coords = outputs2[0]  # 机器人手坐标
            visualizer.update_coordinates(actual_coords, robot_coords)
            visualizer.update_plot()  # 更新绘图
            plt.pause(0.001)  # 短暂停顿以允许GUI更新
            input("按 Enter 键继续...")
        
        dic1 = TIP_dic
        outputs1 = outputs1[:, dic1]
        dic2 = TIP_dic_rb
        outputs2 = outputs2[:, dic2]
        
        # 检查位置数量是否一致
        if outputs1.shape[1] != outputs2.shape[1]:
            raise ValueError(f"位置数量不一致: 文件1有{outputs1.shape[1]}个位置，文件2有{outputs2.shape[1]}个位置")
        
        print(f"文件1数据形状: {outputs1.shape}")
        print(f"文件2数据形状: {outputs2.shape}")
        
        # 截断至较短的时间长度
        min_time_length = min(outputs1.shape[0], outputs2.shape[0])
        outputs1_truncated = outputs1[:min_time_length]
        outputs2_truncated = outputs2[:min_time_length]
        
        # print(f"截断后数据形状 - 文件1: {outputs1_truncated.shape}, 文件2: {outputs2_truncated.shape}")
        
        # 计算每个时间步每个关节的欧几里得距离
        # diff 的形状: [time_length, num_joints, 3] (假设是3D坐标)
        diff = outputs1_truncated - outputs2_truncated  # 形状: [time_length, num_joints, 3] 或 [time_length, num_joints]
        
        if len(diff.shape) == 3:  # 如果是3D坐标 [time_length, num_joints, 3]
            distances = np.sqrt(np.sum(diff**2, axis=2))  # [time_length, num_joints]
            print('3')
        else:  # 如果是1D特征 [time_length, num_joints]
            distances = np.abs(diff)  # [time_length, num_joints]
        
        # 计算每个关节的距离平均值
        mean_distances_per_joint = np.mean(distances, axis=0)  # 在时间维度上求平均
        
        # print("每个关节的距离平均值:")
        # for i, mean_dist in enumerate(mean_distances_per_joint):
        #     print(f"关节 {i}: {mean_dist}")

        print(f"\n所有关节的总平均距离: {np.mean(mean_distances_per_joint)}")
        
        return mean_distances_per_joint
def compare_velocity_difference_mean(file1_path, file2_path, dataset_name='r', dataset_name2='outputs'):
    """
    比较两个HDF5文件中outputs数据集，计算相邻点速度差的绝对值的平均值
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径
        dataset_name: 数据集名称，默认为'r'
        dataset_name2: 第二个数据集名称，默认为'outputs'
    Returns:
        mean_velocity_diff_per_joint: 每个关节的速度差绝对值平均值数组
    """
    # 打开两个HDF5文件
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # 获取outputs数据
        dataset = f1['测试-ceshi']
        outputs1 = dataset[dataset_name][:]
        outputs2 = f2[dataset_name2][:]
        
        outputs1 = outputs1 * scaling  # 将米转换为分米
        
        dic1 = TIP_dic
        outputs1 = outputs1[:, dic1]
        dic2 = TIP_dic_rb
        outputs2 = outputs2[:, dic2]
        
        # 检查位置数量是否一致
        # if outputs1.shape[1] != outputs2.shape[1]:
            # raise ValueError(f"位置数量不一致: 文件1有{outputs1.shape[1]}个位置，文件2有{outputs2.shape[1]}个位置")
        
        # print(f"文件1数据形状: {outputs1.shape}")
        # print(f"文件2数据形状: {outputs2.shape}")
        
        # 截断至较短的时间长度
        min_time_length = min(outputs1.shape[0], outputs2.shape[0])
        outputs1_truncated = outputs1[:min_time_length]
        outputs2_truncated = outputs2[:min_time_length]
        
        # print(f"截断后数据形状 - 文件1: {outputs1_truncated.shape}, 文件2: {outputs2_truncated.shape}")
        
        # 计算相邻点的速度 (相邻帧之间的差值)
        # 速度计算: v[t] = pos[t+1] - pos[t]
        velocities1 = np.diff(outputs1_truncated, axis=0)  # [time_length-1, num_joints, 3] 或 [time_length-1, num_joints]
        velocities2 = np.diff(outputs2_truncated, axis=0)  # [time_length-1, num_joints, 3] 或 [time_length-1, num_joints]
        
        # print(f"速度数据形状 - 文件1: {velocities1.shape}, 文件2: {velocities2.shape}")
        
        # 计算速度差的绝对值
        velocity_diff = np.abs(velocities1 - velocities2)  # [time_length-1, num_joints, ...]
        
        # 计算每个关节速度差绝对值的平均值
        if len(velocity_diff.shape) == 3:  # 如果是3D坐标 [time_length-1, num_joints, 3]
            # 对xyz三个维度都计算差值绝对值的平均
            mean_velocity_diff_per_joint = np.mean(velocity_diff, axis=(0, 2))  # 平均时间维度和空间维度
        else:  # 如果是1D特征 [time_length-1, num_joints]
            mean_velocity_diff_per_joint = np.mean(velocity_diff, axis=0)  # 在时间维度上求平均
        
        # print("每个关节的速度差绝对值平均值:")
        # for i, mean_vel_diff in enumerate(mean_velocity_diff_per_joint):
        #     print(f"关节 {i}: {mean_vel_diff}")

        print(f"\n所有关节的总平均速度差绝对值: {np.mean(mean_velocity_diff_per_joint)}")
        
        return mean_velocity_diff_per_joint
def compare_acceleration_difference_mean(file1_path, file2_path, dataset_name='r', dataset_name2='outputs'):
    """
    比较两个HDF5文件中outputs数据集，计算相邻点加速度差的绝对值的平均值
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径
        dataset_name: 数据集名称，默认为'r'
        dataset_name2: 第二个数据集名称，默认为'outputs'
    Returns:
        mean_acceleration_diff_per_joint: 每个关节的加速度差绝对值平均值数组
    """
    # 打开两个HDF5文件
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # 获取outputs数据
        dataset = f1['测试-ceshi']
        outputs1 = dataset[dataset_name][:]
        outputs2 = f2[dataset_name2][:]
        
        outputs1 = outputs1 * scaling  # 将米转换为分米
        
        dic1 = TIP_dic
        outputs1 = outputs1[:, dic1]
        dic2 = TIP_dic_rb
        outputs2 = outputs2[:, dic2]
        
        # 检查位置数量是否一致
        # if outputs1.shape[1] != outputs2.shape[1]:
            # raise ValueError(f"位置数量不一致: 文件1有{outputs1.shape[1]}个位置，文件2有{outputs2.shape[1]}个位置")
        
        # print(f"文件1数据形状: {outputs1.shape}")
        # print(f"文件2数据形状: {outputs2.shape}")
        
        # 截断至较短的时间长度
        min_time_length = min(outputs1.shape[0], outputs2.shape[0])
        outputs1_truncated = outputs1[:min_time_length]
        outputs2_truncated = outputs2[:min_time_length]
        
        # print(f"截断后数据形状 - 文件1: {outputs1_truncated.shape}, 文件2: {outputs2_truncated.shape}")
        
        # 计算相邻点的速度 (相邻帧之间的差值)
        velocities1 = np.diff(outputs1_truncated, axis=0)  # [time_length-1, num_joints, 3] 或 [time_length-1, num_joints]
        velocities2 = np.diff(outputs2_truncated, axis=0)  # [time_length-1, num_joints, 3] 或 [time_length-1, num_joints]
        
        # 计算相邻点的加速度 (速度的差值)
        # 加速度计算: a[t] = v[t+1] - v[t] = pos[t+2] - 2*pos[t+1] + pos[t]
        accelerations1 = np.diff(velocities1, axis=0)  # [time_length-2, num_joints, 3] 或 [time_length-2, num_joints]
        accelerations2 = np.diff(velocities2, axis=0)  # [time_length-2, num_joints, 3] 或 [time_length-2, num_joints]
        
        # print(f"加速度数据形状 - 文件1: {accelerations1.shape}, 文件2: {accelerations2.shape}")
        
        # 计算加速度差的绝对值
        acceleration_diff = np.abs(accelerations1 - accelerations2)  # [time_length-2, num_joints, ...]
        
        # 计算每个关节加速度差绝对值的平均值
        if len(acceleration_diff.shape) == 3:  # 如果是3D坐标 [time_length-2, num_joints, 3]
            # 对xyz三个维度都计算差值绝对值的平均
            mean_acceleration_diff_per_joint = np.mean(acceleration_diff, axis=(0, 2))  # 平均时间维度和空间维度
        else:  # 如果是1D特征 [time_length-2, num_joints]
            mean_acceleration_diff_per_joint = np.mean(acceleration_diff, axis=0)  # 在时间维度上求平均
        
        # print("每个关节的加速度差绝对值平均值:")
        # for i, mean_acc_diff in enumerate(mean_acceleration_diff_per_joint):
        #     print(f"关节 {i}: {mean_acc_diff}")

        print(f"\n所有关节的总平均加速度差绝对值: {np.mean(mean_acceleration_diff_per_joint)}")
        
        return mean_acceleration_diff_per_joint

# 示例用法
if __name__ == "__main__":
    visualizer = RealTimeVisualizer(actual_connections=hand_connections, robot_connections=robot_connections)
    # ab_experiment_name = 'None'
    # ab_experiment_name = 'ab_vec_loss'
    # ab_experiment_name = 'ab_tip_pos_loss'
    ab_experiment_name = 'ab_col_loss'
    # ab_experiment_name = 'ab_tip_dis_loss'
    file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\test_glove_data0204_aligned.h5"
    file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_output.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_positions_output.h5"
    try:
        mean_distances_per_joint = compare_outputs_distance_mean(
            file1_path, file2_path, 
            dataset_name='r_glove_pos', 
            dataset_name2='joint_positions',
            visualizer=visualizer
        )
        #  新增的速度差比较
        mean_velocity_diff_per_joint = compare_velocity_difference_mean(
            file1_path, file2_path,
            dataset_name='r_glove_pos',
            dataset_name2='joint_positions'
        )
        # 新增的加速度差比较
        mean_acceleration_diff_per_joint = compare_acceleration_difference_mean(
            file1_path, file2_path,
            dataset_name='r_glove_pos',
            dataset_name2='joint_positions'
        )
        # print(f"\n所有关节的距离平均值数组: {mean_distances_per_joint}")
        # for i, mean_dist in enumerate(mean_distances_per_joint):
        #     print(f"关节 {i}: {mean_dist}")
            
    except ValueError as e:
        print(f"错误: {e}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")