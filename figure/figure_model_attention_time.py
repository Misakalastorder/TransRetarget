import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from model.model_poseformer import PoseTransformer
from config.variables_define import *
from matplotlib.ticker import MaxNLocator
def simple_temporal_attention_visualization(model_path, layer_idx=0, heads_to_plot=[0, 1, 2, 3], save_dir=None,
                                          num_frame=3, in_num_joints=None, in_chans=3, out_num_joint=None, out_chans=1,
                                          embed_dim_ratio=64, spatial_depth=6, temporal_depth=2, num_heads=8,
                                          qkv_bias=True, qk_scale=None, drop_path_rate=0.2,
                                          angle_limit_rad=angle_limit_rob):
    """
    简化版本：使用最少的输入来获取注意力权重
    """
    # 创建模型实例
    model = PoseTransformer(
        num_frame=num_frame,             
        in_num_joints=in_num_joints,     
        in_chans=in_chans,               
        out_num_joint=out_num_joint,     
        out_chans=out_chans,             
        embed_dim_ratio=embed_dim_ratio,
        spatial_depth=spatial_depth,
        temporal_depth=temporal_depth,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop_path_rate=drop_path_rate,
        angle_limit_rad=angle_limit_rad
    )
    
    # 加载预训练权重（参考main_train.py的加载方式）
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model.eval()
    print(f"Model loaded from {model_path}")

    # 创建简单的输入（只需正确维度）
    embed_dim = embed_dim_ratio * in_num_joints
    simple_input = torch.randn(1, num_frame, embed_dim)  # [batch, frames, embed_dim]

    # 捕获注意力权重
    attention_weights_list = []
    original_attn_forward = model.blocks[layer_idx].attn.forward

    def capture_attention_forward(self, x):
        B, N, C = x.shape  # N = num_frame (frames)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attention_weights_list.append(attn.detach().cpu())

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # 替换forward方法
    bound_method = capture_attention_forward.__get__(model.blocks[layer_idx].attn, type(model.blocks[layer_idx].attn))
    model.blocks[layer_idx].attn.forward = bound_method

    with torch.no_grad():
        # 直接调用时序特征提取部分
        result = model.forward_features(simple_input)

    # 恢复原始方法
    model.blocks[layer_idx].attn.forward = original_attn_forward

    # 可视化捕获的注意力权重
    if attention_weights_list:
        attn_weights = attention_weights_list[0][0]  # [num_heads, frames, frames]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for i, head_idx in enumerate(heads_to_plot):
            if i >= 4:
                break

            head_attn = attn_weights[head_idx]  # [frames, frames]
            # 归一化
            head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min() + 1e-8)

            im = axes[i].imshow(head_attn.numpy(), cmap='RdYlBu_r', aspect='auto')
            # 设置坐标轴刻度标签的字体大小
            axes[i].tick_params(axis='both', which='major', labelsize=14)  # 调整坐标数字字号
            axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[i].yaxis.set_major_locator(MaxNLocator(integer=True))
            # axes[i].set_title(f'Temporal Head {head_idx}')
            # axes[i].set_xlabel('Key Frames')
            # axes[i].set_ylabel('Query Frames')
            # plt.colorbar(im, ax=axes[i])

        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'temporal_layer_{layer_idx}_simple_attention.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Simple temporal attention visualization saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # 使用示例
    model_path = "D:\\2026\\code\\TransHandR\\TransHandR\\checkpoint\\models\\new0204\\linker\\model_final.pth"
    model_config = {
        'num_frame': receptive_field,
        'in_num_joints': num_joints,
        'in_chans': in_chans,
        'out_num_joint': out_num_joint,
        'out_chans': 1,
        'embed_dim_ratio': embed_dim_ratio,
        'spatial_depth': spatial_depth,
        'temporal_depth': temporal_depth,
        'num_heads': num_heads,
        'qkv_bias': qkv_bias,
        'qk_scale': qk_scale,
        'drop_path_rate': drop_path_rate,
        'angle_limit_rad': angle_limit_rob
    }

    # 简化版可视化（使用最小输入）
    simple_temporal_attention_visualization(
        model_path=model_path,
        layer_idx=0,  # 时序Transformer层索引
        heads_to_plot=[0, 1, 2, 3],
        save_dir="./attention_plots",
        **model_config
    )
    
    print("Temporal attention visualization functions are ready!")