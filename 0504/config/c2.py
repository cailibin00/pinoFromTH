"""
配置 C2 — SiLU 激活 + 残差连接
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reynold_pinn import Config as BaseConfig


class Config(BaseConfig):
    # ========== 激活函数 ==========
    Act = "silu"             # ← tanh → silu

    # ========== 网络架构 ==========
    use_residual = True       # ← 开启残差连接

    # ========== 输出目录 ==========
    output_dir = "output_c2_silu_res"
