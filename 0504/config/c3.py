"""
配置 C3 — SiLU + 深网络
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reynold_pinn import Config as BaseConfig


class Config(BaseConfig):
    # ========== 激活函数 ==========
    Act = "silu"

    # ========== 网络架构 ==========
    use_residual = True
    layer_sizes = [2, 256, 256, 256, 128, 128, 2]  # 更宽

    # ========== 输出目录 ==========
    output_dir = "output_c3_silu_deep"
