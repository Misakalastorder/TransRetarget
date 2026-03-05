# 对比文件1 output/comparing/linker_visionpro.h5
# 对比文件2 output/h5/linker/linker_output.h5

import h5py
import numpy as np
from config.variables_define import *
from model.loss import RealTimeVisualizer
import matplotlib.pyplot as plt
def compare_outputs_mse_per_joint(file1_path, file2_path, dataset_name='r',dataset_name2='outputs', visualizer=None):
    """
    比较两个HDF5文件中outputs数据集的角度数组的均方差，分别计算每个关节的均方差
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径
        dataset_name: 数据集名称，默认为'outputs'
        dataset_name2: 第二个数据集名称，默认为'outputs'
        visualizer: 实时可视化器实例
    Returns:
        mse_per_joint: 每个关节的均方差数组
    """
    # 打开两个HDF5文件
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # 获取outputs数据
        # key = "S000079_P0008"
        # '测试-ceshi'
        dataset = f1['测试-ceshi']
        # dataset = f1['S000079_P0008']
        outputs1 = dataset[dataset_name][:]
        outputs2 = f2[dataset_name2][:]
        outputs1 = outputs1 * 4.5/4.0  # slahmr ours可用
        # outputs1 = outputs1 * 4.7/4.0  # 将米转换为分米
        # outputs2 = outputs2 * 10  # 将米转换为分米
        if visualizer is not None:
        # 只显示批次中的第一个样本
        # print("角度格式", predicted_angle_for_fk.shape)
        # print(predicted_angle_for_fk[0])
            actual_coords = outputs1[0]  # 实际手部坐标
            robot_coords = outputs2[0]  # 机器人手坐标
            # print("实际手部坐标", actual_coords)
            # print("机器人手坐标", robot_coords)
            visualizer.update_coordinates(actual_coords, robot_coords)
            visualizer.update_plot()  # 更新绘图
            plt.pause(0.001)  # 短暂停顿以允许GUI更新
            #等待确认后关闭
            input("按 Enter 键继续...")
        dic1= TIP_dic
        outputs1 = outputs1[:,dic1]
        dic2= TIP_dic_rb
        outputs2 = outputs2[:,dic2]
        # 检查角度数量是否一致
        if outputs1.shape[1] != outputs2.shape[1]:
            raise ValueError(f"位置数量不一致: 文件1有{outputs1.shape[1]}个位置，文件2有{outputs2.shape[1]}个角度")
        
        print(f"文件1数据形状: {outputs1.shape}")
        print(f"文件2数据形状: {outputs2.shape}")
        
        # 截断至较短的时间长度
        min_time_length = min(outputs1.shape[0], outputs2.shape[0])
        outputs1_truncated = outputs1[:min_time_length]
        outputs2_truncated = outputs2[:min_time_length]
        
        print(f"截断后数据形状 - 文件1: {outputs1_truncated.shape}, 文件2: {outputs2_truncated.shape}")
        
        # 计算每个关节的均方差
        diff = outputs1_truncated - outputs2_truncated  # 形状: [time_length, num_joints,3]
        #取模长
        diff = np.linalg.norm(diff, axis=2)
        mse_per_joint = np.mean(diff, axis=0)  # 在时间维度上求平均，保留关节维度
        
        print("每个关节的均方差:")
        for i, mse in enumerate(mse_per_joint):
            # print(f"关节 {i}: {mse}")
            print(f"{mse}")

        print(f"\n平均总体均方差: {np.mean(mse_per_joint)}")
        rmse = np.sqrt(mse_per_joint)
        print(f"\n均方根误差: {rmse}")
        return mse_per_joint

# 示例用法
if __name__ == "__main__":
    visualizer = RealTimeVisualizer(actual_connections=hand_connections, robot_connections=robot_connections)
    # ab_experiment_name = 'None'
    # ab_experiment_name = 'ab_vec_loss'
    # ab_experiment_name = 'ab_tip_pos_loss'
    ab_experiment_name = 'ab_col_loss'
    # ab_experiment_name = 'ab_tip_dis_loss'
    # 替换为实际的文件路径
    file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\test_glove_data0204_aligned.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_output.h5"
    file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{ab_experiment_name}\\{hand_brand}\\{hand_brand}_positions_output.h5"
    # file1_path = "D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5\\sign_glove_aligned.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_positions_slahmr_output2.h5"
    # all_frames_data = load_human_tip_from_h5_all_frames(h5_file_path, subject_key="S000079_P0008")
    # key = "S000079_P0008"
    try:
        mse_per_joint = compare_outputs_mse_per_joint(file1_path, file2_path, dataset_name='r_glove_pos', dataset_name2='joint_positions',visualizer=visualizer)
        # mse_per_joint = np.mean(mse_per_joint, axis=1)
        
        # # print(f"\n所有关节的均方差数组: {mse_per_joint}")
        # for i, mse in enumerate(mse_per_joint):
        #     print(f"{mse}")
    except ValueError as e:
        print(f"错误: {e}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")