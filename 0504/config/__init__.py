"""
配置注册中心 — 按序号加载实验配置。
用法:
    from config import get_config
    cfg = get_config(1)   # 使用 c1.py
    cfg = get_config(2)   # 使用 c2.py

添加新配置: 复制 c1.py → cN.py，修改类属性即可。
"""

import importlib


def get_config(config_id=1):
    """
    加载指定序号的配置。

    Args:
        config_id: 配置编号 (1, 2, 3, ...)，对应 config/c1.py, c2.py, ...

    Returns:
        Config 实例
    """
    module_name = f"config.c{config_id}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise FileNotFoundError(
            f"配置文件 {module_name}.py 不存在！"
            f"请在 config/ 目录下创建 c{config_id}.py"
        )
    return mod.Config()
