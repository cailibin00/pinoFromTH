"""
配置 C1 — 基准配置 (tanh + 原版网络)
与原始 Config 完全一致，作为对照实验。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reynold_pinn import Config as BaseConfig


class Config(BaseConfig):
    """完全继承基类配置，不做任何覆写。"""
    pass
