# 对比文件1 output/comparing/linker_visionpro.h5
# 对比文件2 output/h5/linker/linker_output.h5

import h5py
import numpy as np
from config.variables_define import *

def compare_outputs_mse_per_joint(file1_path, file2_path, dataset_name='outputs'):
    """
    比较两个HDF5文件中outputs数据集的角度数组的均方差，分别计算每个关节的均方差
    
    Args:
        file1_path: 第一个HDF5文件路径
        file2_path: 第二个HDF5文件路径
        dataset_name: 数据集名称，默认为'outputs'
    
    Returns:
        mse_per_joint: 每个关节的均方差数组
    """
    # 打开两个HDF5文件
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        # 获取outputs数据
        outputs1 = f1[dataset_name][:]
        outputs2 = f2[dataset_name][:]
        
        # 检查角度数量是否一致
        if outputs1.shape[1] != outputs2.shape[1]:
            raise ValueError(f"角度数量不一致: 文件1有{outputs1.shape[1]}个角度，文件2有{outputs2.shape[1]}个角度")
        
        print(f"文件1数据形状: {outputs1.shape}")
        print(f"文件2数据形状: {outputs2.shape}")
        
        # 截断至较短的时间长度
        min_time_length = min(outputs1.shape[0], outputs2.shape[0])
        outputs1_truncated = outputs1[:min_time_length]
        outputs2_truncated = outputs2[:min_time_length]
        
        print(f"截断后数据形状 - 文件1: {outputs1_truncated.shape}, 文件2: {outputs2_truncated.shape}")
        
        # 计算每个关节的均方差
        diff = outputs1_truncated - outputs2_truncated  # 形状: [time_length, num_joints]
        mse_per_joint = np.mean(diff ** 2, axis=0)  # 在时间维度上求平均，保留关节维度
        
        print("每个关节的均方差:")
        for i, mse in enumerate(mse_per_joint):
            # print(f"关节 {i}: {mse}")
            print(f"{mse}")
        
        print(f"\n平均总体均方差: {np.mean(mse_per_joint)}")
        
        return mse_per_joint

# 示例用法
if __name__ == "__main__":
    # 替换为实际的文件路径
    file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_visionpro.h5"
    # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output1.h5"
    file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output1.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_output2.h5"
    # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_slahmr.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output.h5"
    # # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output1.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\slahmr\\{hand_brand}\\{hand_brand}_slahmr_output2.h5"
    # file1_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\comparing\\{hand_brand}_visionpro10000.h5"
    # file2_path = f"D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\{hand_brand}\\{hand_brand}_outputVP2.h5"
    try:
        mse_per_joint = compare_outputs_mse_per_joint(file1_path, file2_path)
        print(f"\n所有关节的均方差数组: {mse_per_joint}")
    except ValueError as e:
        print(f"错误: {e}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")