"""
figure_loss_real_rb
绘制真实数据与模型预测数据之间的损失
输入为两个h5文件，分别包含真实数据和预测数据
输出为一个h5文件，包含损失数据
保存至/output/loss/real_rb_loss.h5
"""
from config.variables_define import *
import numpy as np
import h5py
import os
import torch
import torch.nn.functional as F

def h5_to_numpy(h5_file_path,key='data'):
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file[key]
        data_out = data['r_glove_pos']  # 假设数据存储在'r_global_pos'键下
        data_out = np.array(data_out)
        return data_out
    
def h5_to_numpy_pred(h5_file_path,key='data'):
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file[key]
        data = np.array(data)
        return data

def numpy_to_h5(numpy_array, h5_file_path, key='data'):
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset(key, data=numpy_array)

def calculate_joint_angle_rb(pred_data):
    PIP = PIP_dic_rb
    MCP = MCP_dic_rb
    TIP = TIP_dic_rb
    pred_data_5 = np.repeat(pred_data[:, 0:1, :], 5, axis=1)
    vec1 = pred_data[:, TIP, :] - pred_data[:, MCP, :]
    vec2 = pred_data[:, MCP, :] - pred_data_5
    angle = np.arccos(np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1)))
    return angle
def calculate_joint_angle_real(real_data):
    '''
    计算特定关节的角度
    输入：real_data，形状为(num_samples, num_joints, 3)
    输出：angle，形状为(num_samples, num_angles)
    '''
    ### 适用于visionpro采集数据
    Pip = source_dic['PIP_dic']  # 近端关节索引
    Tip = source_dic['TIP_dic']  # 指尖关节索引
    Pip[0] = Pip[0] + 1  # 大拇指PIP关节索引调整
    MCP_dic = source_dic['MCP_dic'] # 掌指关节索引
    MCP_dic[0] = MCP_dic[0] + 1

    Palm = source_dic['PALM_dic'] # 手掌根部关节索引
    # 将real_data[:, 0, :]从[f,3]复制拓展为[f,5,3]
    # 正确创建重复的数组：取第1个关节并扩展为(num_samples, 5, 3)
    real_data_5 = np.repeat(real_data[:, 0:1, :], 5, axis=1)  # 使用repeat
    vec1 = real_data[:, Tip, :] - real_data[:, MCP_dic, :]
    vec2 = real_data[:, MCP_dic, :] - real_data_5

    ### 适用于slahmr采集数据
    # Pip = source_dic['PIP_dic']  # 近端关节索引
    # Tip = source_dic['TIP_dic']  # 指尖关节索引
    # MCP_dic = source_dic['MCP_dic'] # 掌指关节索引
    # real_data_5 = np.repeat(real_data[:, 0:1, :], 5, axis=1)  # 使用repeat
    # vec1 = real_data[:, Tip, :] - real_data[:, MCP_dic, :]
    # vec2 = real_data[:, MCP_dic, :] - real_data_5
    # 计算夹角
    angle = np.arccos(np.clip(np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1) + 1e-8), -1.0, 1.0))
    return angle
def calculate_loss(real_data, pred_angle,hand_fk,device):
    # 输入的真实数据应该是三维数组，shape为(num_samples, num_joints, 3)
    # 输入的预测数据应该是三维数组，shape为(num_samples, out_num_joints, 1)
    # 要借助前向运动学将预测数据即角度数据转换为与真实数据相同的格式即三维坐标
    # 为predicted_angle加五列零 以补足指尖固定关节的角度
    # 转为torch张量
    pred_angle = torch.tensor(pred_angle, dtype=torch.float32).to(device)
    pred_angle = F.pad(pred_angle, (0, 5, 0, 0), "constant", 0)
    print(pred_angle.shape)
    
    # 前向运动学计算3D关键点位置
    positions, orientations, pred_data = hand_fk.forward(pred_angle)
    pred_data = pred_data.cpu().numpy()
    #输出格式
    print(pred_data.shape)
    print(real_data.shape)
    
    real_angle = calculate_joint_angle_real(real_data)
    print(real_angle.shape)
    rb_angle = calculate_joint_angle_rb(pred_data)
    print(rb_angle.shape)
    # 若序列不等长，截断
    if real_angle.shape[0] != rb_angle.shape[0]:
        rb_angle = rb_angle[:real_angle.shape[0]]
        real_angle = real_angle[:rb_angle.shape[0]]
    print('截断后',rb_angle.shape)
    # 对上述两个角度求均方差
    losses = ((real_angle - rb_angle) ** 2).mean(axis=0)
    return losses

def main(real_data_path, pred_data_path, loss_data_path, hand_fk,input_key='data', output_key='loss',device='cpu'):
    real_data = h5_to_numpy(real_data_path,key=input_key)
    pred_data = h5_to_numpy_pred(pred_data_path,key=output_key)
    losses = calculate_loss(real_data, pred_data, hand_fk,device=device)
    numpy_to_h5(losses, loss_data_path,key='loss')

from model.loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hand_fk = create_hand_kinematics(urdf_file, hand_cfg, device, scale_factor=scaling_factor_rb,axis_correction_matrix=correction_matrix)
if __name__ == '__main__':
    DATA_PATH = 'D:\\2026\\code\\TransHandR\\TransHandR\\dataset\\data\\h5'
    OUTPUT_PATH = 'D:\\2026\\code\\TransHandR\\TransHandR\\output\\h5\\linker'
    LOSS_PATH = 'D:\\2026\\code\\TransHandR\\TransHandR\\output\\loss'
    # 输入真实数据与预测数据
    real_data_path = os.path.join(DATA_PATH, 'glove_data_aligned.h5')
    pred_data_path = os.path.join(OUTPUT_PATH, 'linker_output.h5')
    # 输出损失数据
    loss_data_path = os.path.join(LOSS_PATH, 'real_rb_loss.h5')
    main(real_data_path, pred_data_path, loss_data_path,hand_fk,input_key='测试-ceshi', output_key='outputs',device=device)

