"""
检查 TF checkpoint 的模型架构 & 生成 model_config.json。
用法:
    # 仅查看 checkpoint 结构
    python inspect_checkpoint.py output_resF_silu_small_epoch30000/checkpoints/epochs_best_model
    # 查看并自动生成 model_config.json
    python inspect_checkpoint.py output_resF_silu_small_epoch30000/checkpoints/epochs_best_model --save-config
"""

import os
import json
import argparse
from tensorflow.python.training import py_checkpoint_reader


def inspect_ckpt(ckpt_prefix):
    """读取 TF checkpoint 并打印各变量的 name 和 shape。"""
    if not os.path.exists(ckpt_prefix + ".index"):
        candidates = [f for f in os.listdir(os.path.dirname(ckpt_prefix) or '.')
                      if '.data-00000-of-00001' in f] if os.path.exists(os.path.dirname(ckpt_prefix) or '.') else []
        if not candidates:
            # Try looking for .index files
            d = os.path.dirname(ckpt_prefix) or '.'
            if os.path.exists(d):
                ckpt_files = [f for f in os.listdir(d) if f.endswith('.index')]
                if ckpt_files:
                    ckpt_prefix = os.path.join(d, ckpt_files[0].replace('.index', ''))
        if not os.path.exists(ckpt_prefix + ".index"):
            print(f"ERROR: 找不到 checkpoint: {ckpt_prefix}")
            return None, None

    reader = py_checkpoint_reader.NewCheckpointReader(ckpt_prefix)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()

    print(f"\nCheckpoint: {ckpt_prefix}")
    print(f"共 {len(var_to_shape_map)} 个变量\n")

    # 提取 Dense (kernel/bias) 层
    dense_layers = {}  # layer_name → {kernel_shape, bias_shape}
    other_vars = []

    for name in sorted(var_to_shape_map.keys()):
        shape = var_to_shape_map[name]
        dtype = var_to_dtype_map.get(name, "?")
        # 跳过 optimizer 变量
        if 'Adam' in name or 'beta' in name or 'learning_rate' in name or 'iter' in name or 'counted' in name:
            continue
        parts = name.split('/')
        layer_name = parts[0] if len(parts) > 1 else 'root'

        if 'kernel' in name.lower() and len(shape) == 2:
            in_dim, out_dim = int(shape[0]), int(shape[1])
            if layer_name not in dense_layers:
                dense_layers[layer_name] = {}
            dense_layers[layer_name]['kernel'] = (in_dim, out_dim)
        elif 'bias' in name.lower() and len(shape) == 1:
            out_dim = int(shape[0])
            if layer_name not in dense_layers:
                dense_layers[layer_name] = {}
            dense_layers[layer_name]['bias'] = out_dim
        else:
            other_vars.append((name, shape, dtype))

    # 打印 Dense 层
    print(f"{'Layer':<30s} {'Kernel':<15s} {'Bias':<12s}")
    print("-" * 60)

    sorted_layers = sorted(dense_layers.items(),
                           key=lambda x: x[0].lower().replace('dense_', '').zfill(5))
    layer_sizes = []
    for layer_name, info in sorted_layers:
        k_shape = info.get('kernel', ('?', '?'))
        b_shape = info.get('bias', '?')
        print(f"  {layer_name:<28s} {str(k_shape):<15s} {str(b_shape):<12s}")
        if isinstance(k_shape, tuple) and len(k_shape) == 2:
            if not layer_sizes:
                layer_sizes.append(k_shape[0])
            layer_sizes.append(k_shape[1])

    # 其他变量
    if other_vars:
        print(f"\n--- 其他变量 ({len(other_vars)}) ---")
        for name, shape, dtype in other_vars[:20]:
            print(f"  {name:<50s} {str(shape):<20s} {str(dtype):<10s}")
        if len(other_vars) > 20:
            print(f"  ... 还有 {len(other_vars) - 20} 个变量")

    # 推断 layer_sizes
    if layer_sizes:
        print(f"\n推断的 layer_sizes: {layer_sizes}")
    else:
        layer_sizes = [2, 2]  # fallback

    # 推断 coslayer_mode
    has_para_exp_bc = any('para_exp_BC' in name for name, _, _ in other_vars)
    has_fc_paths = any('r_fc' in name or 'theta_fc' in name for name, _, _ in other_vars)
    has_phy_cos = any('phy_cos' in name for name, _, _ in other_vars)

    if has_para_exp_bc:
        coslayer_mode = "simple"
    elif has_fc_paths or has_phy_cos:
        coslayer_mode = "mlp"
    else:
        coslayer_mode = "simple"

    print(f"推断的 coslayer_mode: {coslayer_mode}")
    print(f"  (has_para_exp_BC={has_para_exp_bc}, has_fc_paths={has_fc_paths}, has_phy_cos={has_phy_cos})")

    config = {
        'layer_sizes': [int(x) for x in layer_sizes],
        'core': 'MLP',
        'Act': '?',
        'use_residual': False,
        'coslayer_mode': coslayer_mode,
        'output_head_dim': 64,
        'kan_grid_size': 5,
        'kan_spline_order': 3,
        'pikan_layer_sizes': [2, 64, 64, 64, 64, 2],
    }

    return var_to_shape_map, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TF checkpoint architecture")
    parser.add_argument("ckpt_path", help="Path to checkpoint (without .index)")
    parser.add_argument("--save-config", action="store_true",
                        help="Save inferred model_config.json next to the checkpoint")
    args = parser.parse_args()

    var_map, config = inspect_ckpt(args.ckpt_path)

    if args.save_config and config:
        out_path = os.path.join(os.path.dirname(args.ckpt_path), 'model_config.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"\n✓ model_config.json 已保存到: {out_path}")
        print(f"  请检查并手动修正 Act (激活函数) 等无法从 checkpoint 推断的参数。")
