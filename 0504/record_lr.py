"""
学习率曲线记录 & 绘图工具
==========================
用法:
    python record_lr.py 2                        # 按 config/c2.py 计算LR，写入对应 output 目录
    python record_lr.py 2 --plot-only            # 仅画图 (JSON中已有 lr_history)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 各配置的关键参数硬编码 (避免 import tensorflow) ───────────────────────────
# 与 config/cN.py 保持一致，如需新增配置请在此添加
CONFIG_PARAMS = {
    1: dict(peak_lr=1e-3, warmup_epochs=1000, total_epochs=30000, min_lr=1e-6,
            output_dir="output_output_resF_tanh_largec1_tanh"),
    2: dict(peak_lr=1e-3, warmup_epochs=1000, total_epochs=30000, min_lr=1e-6,
            output_dir="output_resF_silu_small"),
    3: dict(peak_lr=1e-3, warmup_epochs=1000, total_epochs=30000, min_lr=1e-6,
            output_dir="output_c3_silu_wide"),
}


def get_config_params(config_id):
    """获取指定配置的关键参数，优先从 config 模块读取，失败则用硬编码"""
    try:
        # 尝试直接解析 cN.py 文件，避免触发 reynold_pinn 的 import
        config_path = os.path.join(SCRIPT_DIR, "config", f"c{config_id}.py")
        params = {}
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 简单解析 Python 赋值语句
        for line in content.split('\n'):
            line = line.strip()
            for key in ['peak_lr', 'warmup_epochs', 'total_epochs', 'min_lr',
                        'output_dir']:
                if line.startswith(f'{key} ') or line.startswith(f'{key}='):
                    # 提取 = 右边的值
                    val_str = line.split('=', 1)[1].strip().rstrip('#').strip()
                    try:
                        # 尝试 eval 数字
                        val = eval(val_str)
                    except Exception:
                        val = val_str.strip('"').strip("'")
                    params[key] = val
        if 'peak_lr' in params:
            return params
    except Exception:
        pass
    # 回退到硬编码
    if config_id in CONFIG_PARAMS:
        return CONFIG_PARAMS[config_id]
    raise ValueError(f"找不到配置 C{config_id}，请在 CONFIG_PARAMS 中添加或检查 config/c{config_id}.py")


def compute_lr_curve(peak_lr, warmup_epochs, total_epochs, min_lr=1e-6,
                     record_every=10):
    """
    复现 WarmupCosineDecay 的学习率曲线。
    返回 (epochs, lrs) 列表。
    """
    epochs = []
    lrs = []
    for step in range(1, total_epochs + 1):
        if step < warmup_epochs:
            lr = peak_lr * (step / max(warmup_epochs, 1))
        else:
            decay_steps = total_epochs - warmup_epochs
            cosine = 0.5 * (1.0 + np.cos(
                np.pi * (step - warmup_epochs) / max(decay_steps, 1)
            ))
            lr = min_lr + (peak_lr - min_lr) * cosine
        if step % record_every == 0 or step == 1:
            epochs.append(step)
            lrs.append(lr)
    return epochs, lrs


def plot_lr(epochs, lrs, save_path, peak_lr, min_lr, warmup_epochs, total_epochs,
            dpi=300):
    """绘制学习率曲线 (线性坐标 + 对数坐标 双图)"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=150)
    fig.suptitle(f"Learning Rate Schedule\n"
                 f"peak={peak_lr:.0e}, min={min_lr:.0e}, "
                 f"warmup={warmup_epochs}, total={total_epochs}",
                 fontsize=16)

    # ── 上：线性坐标 ──
    ax = axes[0]
    ax.plot(epochs, lrs, 'b-', lw=0.8)
    ax.axvline(warmup_epochs, color='gray', ls='--', lw=1, alpha=0.7,
               label=f'warmup end ({warmup_epochs})')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Linear Scale', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # ── 下：对数坐标 ──
    ax = axes[1]
    ax.semilogy(epochs, lrs, 'b-', lw=0.8)
    ax.axvline(warmup_epochs, color='gray', ls='--', lw=1, alpha=0.7,
               label=f'warmup end ({warmup_epochs})')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Learning Rate', fontsize=14)
    ax.set_title('Log Scale', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[Plot] LR 图保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="学习率曲线记录 & 绘图")
    parser.add_argument("config_id", nargs="?", type=int, default=2,
                        help="配置序号 (对应 config/cN.py)")
    parser.add_argument("--plot-only", action="store_true",
                        help="仅画图，不重新计算 LR")
    parser.add_argument("--dpi", type=int, default=300, help="图片 DPI")
    args = parser.parse_args()

    cfg = get_config_params(args.config_id)
    output_dir = os.path.join(SCRIPT_DIR, cfg['output_dir'])
    figures_dir = os.path.join(output_dir, "figures")
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(figures_dir, exist_ok=True)

    json_path = os.path.join(log_dir, "loss_history.json")
    peak_lr = cfg['peak_lr']
    warmup_epochs = cfg['warmup_epochs']
    total_epochs = cfg['total_epochs']
    min_lr = cfg['min_lr']

    if not args.plot_only:
        # ── 计算 LR 曲线 ──────────────────────────────────────────────────
        print(f"[LR] 按 C{args.config_id} 计算学习率曲线: "
              f"peak={peak_lr}, warmup={warmup_epochs}, "
              f"total={total_epochs}, min={min_lr}")
        epochs, lrs = compute_lr_curve(
            peak_lr=peak_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
            record_every=10,
        )

        # ── 回写 JSON ─────────────────────────────────────────────────────
        with open(json_path, 'r') as f:
            data = json.load(f)
        data['lr_history'] = lrs
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[LR] lr_history ({len(lrs)} 点) 已写入: {json_path}")
    else:
        # ── 从 JSON 读取已有 lr_history ────────────────────────────────────
        with open(json_path, 'r') as f:
            data = json.load(f)
        lrs = data.get('lr_history')
        if lrs is None:
            print("[ERROR] JSON 中没有 lr_history，请先不带 --plot-only 运行")
            sys.exit(1)
        total = len(lrs) * 10
        epochs = list(range(10, total + 1, 10))
        print(f"[LR] 从 JSON 读取 lr_history ({len(lrs)} 点)")

    # ── 画图 ──────────────────────────────────────────────────────────────────
    fig_path = os.path.join(figures_dir, "learning_rate_schedule.png")
    plot_lr(epochs, lrs, fig_path, peak_lr, min_lr,
            warmup_epochs, total_epochs, dpi=args.dpi)

    print("完成!")


if __name__ == "__main__":
    main()
