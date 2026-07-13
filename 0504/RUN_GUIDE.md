# 0504 原版 TensorFlow PINN 运行指南（Ubuntu Linux）

## 0. 前提：将项目传到 Linux 服务器

```bash
# 从 Windows 用 scp 传整个 0504 文件夹到 Linux
# 在 Windows PowerShell 中执行：
scp -r E:\a_lab\lab_Pro\pinoFromTH\0504 user@your-server:/home/user/pinoFromTH/

# 或者用 rsync（Linux 之间）：
rsync -av /path/to/pinoFromTH/0504/ user@your-server:/home/user/pinoFromTH/0504/
```

以下假设项目位于 `/home/user/pinoFromTH/0504`，请替换为实际路径。

---

## 目录结构

```
0504/
├── reynold_pinn.py                  # ★ 主训练脚本（入口）
├── compare_fem_pinn_final.py        # 评估脚本（FEM vs PINN 对比）
├── tensordiffeq/                    # 自定义 PINN 框架
│   ├── __init__.py
│   ├── models.py                    # CollocationSolverND 核心类
│   ├── fit.py                       # 训练循环 + PCGrad + 自适应权重
│   ├── networks.py                  # 网络架构（Coslayer, Gate, BC层）
│   ├── boundaries.py                # 边界条件定义
│   ├── domains.py                   # 计算域 + 配点生成
│   ├── utils.py                     # MSE, LHS采样, 权重设置
│   ├── optimizers.py                # L-BFGS 优化器
│   ├── helpers.py                   # L2误差工具
│   ├── output.py                    # 打印输出
│   ├── sampling.py                  # 拉丁超立方采样
│   └── PCGrad.py                    # PCGrad 辅助
├── p_FBNS.txt                       # FEM 参考压力数据 (401×401=160801行)
├── g_FBNS.txt                       # FEM 参考空化率数据
├── epochs_best_model.data-00000-of-00001  # 预训练权重
├── epochs_best_model.index          # 预训练权重索引
├── comparison_results/              # 评估输出（图片 + metrics.txt）
│   ├── metrics.txt                  # P_rel_L2=1.39e-2 等指标
│   ├── fig1a_P_fem.png ~ fig8_jfo_complement.png
│   └── ...
├── reynolds_pinn_N4900_iter80000/   # 完整模型 SavedModel
│   ├── saved_model.pb
│   ├── keras_metadata.pb
│   └── variables/
└── results/                         # 额外权重副本
```

---

## 依赖环境

### 核心依赖

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| Python | **3.10** | 从 `__pycache__/*.cpython-310.pyc` 确认 |
| tensorflow | **2.13.0**（推荐）或 **2.10~2.15** | 代码混用 TF1/2 API |
| tensorflow-probability | **0.20.0**（匹配 TF 2.13） | L-BFGS 需要 `tfp.optimizer.lbfgs_minimize` |
| numpy | 任意 | 数值计算 |
| scipy | 任意 | RAR 方法中 `griddata` 插值 |
| matplotlib | 任意 | 可视化 |
| tqdm | 任意 | 训练进度条 |
| pyDOE2 | 任意 | 拉丁超立方采样 |
| pyfiglet | 任意 | ASCII Art 输出（可选） |

### GPU 支持

| 组件 | TF 2.10 ~ 2.12 | TF 2.13 ~ 2.14 | TF 2.15 |
|------|----------------|----------------|---------|
| CUDA | 11.2 ~ 11.8 | 11.8 | 12.2 |
| cuDNN | 8.1 ~ 8.6 | 8.6 | 8.9 |
| NVIDIA 驱动 | >= 525.60.13 | >= 525.60.13 | >= 525.60.13 |

> **推荐 TF 2.13**：`tf.gradients` 仍可用，GPU 支持稳定，Python 3.10 完全兼容。

如果不用 GPU，跳过 CUDA/cuDNN 安装，pip 会自动装 CPU 版 TF。

---

## 完整安装步骤

### 步骤 1：安装 NVIDIA 驱动 + CUDA + cuDNN（仅 GPU）

**方式 A：用 conda 一键搞定（推荐，最省事）**

```bash
# conda 会自动安装匹配的 CUDA 和 cuDNN
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.6
```

**方式 B：系统级 apt 安装**

```bash
# 安装 NVIDIA 驱动（如果还没装）
sudo apt update
sudo apt install nvidia-driver-535

# 安装 CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 配置环境变量（追加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 安装 cuDNN 8.6（从 NVIDIA 官网下载后）
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.*.deb
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev
```

> 重启后再继续下一步。

### 步骤 2：创建 Python 3.10 虚拟环境

```bash
# 用 conda
conda create -n tf0504 python=3.10 -y
conda activate tf0504

# 或者用 venv（系统 Python 3.10）
python3.10 -m venv venv_tf0504
source venv_tf0504/bin/activate
```

### 步骤 3：安装 TensorFlow

```bash
# GPU 版（推荐，约 30 分钟完成训练）
pip install tensorflow==2.13.0

# 或者 CPU 版（非常慢，约 4-8 小时）
pip install tensorflow-cpu==2.13.0
```

### 步骤 4：安装 tensorflow-probability

```bash
# TF 2.13 → tfp 0.20
pip install tensorflow-probability==0.20.0

# 其他版本对应关系：
# TF 2.10.x → pip install tensorflow-probability==0.18.0
# TF 2.11.x → pip install tensorflow-probability==0.19.0
# TF 2.12.x → pip install tensorflow-probability==0.19.0
# TF 2.14.x → pip install tensorflow-probability==0.22.0
# TF 2.15.x → pip install tensorflow-probability==0.22.0
```

### 步骤 5：安装其余依赖

```bash
pip install numpy scipy matplotlib tqdm pyDOE2 pyfiglet
```

### 步骤 6：验证安装

```bash
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
"
```

预期输出：
```
TF version: 2.13.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 运行训练

### 从头训练

```bash
cd /home/user/pinoFromTH/0504
python reynold_pinn.py
```

训练参数（在 `reynold_pinn.py` 的 `Config` 类中）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_f` | 4900 | 配点数 |
| `N_train` | 5000 | 每轮迭代数 |
| `NL_train` | 4 | RAD 细化轮数 |
| `layer_sizes` | `[2, 128, 128, 128, 128, 2]` | 网络结构 |
| 训练总量 | `5000 × 4 × 4 = 80000` | 4 阶段 × 4 轮迭代 |
| 优化 | PCGrad + 自适应损失权重 | `PCGrad_true=True` |

**输出文件**：
- `epochs_best_model.data-00000-of-00001` / `.index` — 最佳模型权重
- `reynolds_pinn_N4900_iter80000/` — 完整 SavedModel
- `reynolds_pinn_N4900_iter80000_*.png` — 训练结果图

训练完成后自动生成 4 张图：
- `*_pressure_contour.png` — 压力分布云图
- `*_cavitation_contour.png` — 空化率分布云图
- `*_H_only.png` — 膜厚分布图
- `*_loss_log.png` — 训练损失曲线（对数坐标）

### 后台运行（避免 SSH 断开中断训练）

```bash
# 用 nohup 在后台跑，输出写到 train.log
cd /home/user/pinoFromTH/0504
nohup python -u reynold_pinn.py > train.log 2>&1 &

# 查看进度
tail -f train.log

# 查看后台任务
jobs -l
```

### 使用预训练权重直接评估（不重新训练）

```bash
cd /home/user/pinoFromTH/0504
python compare_fem_pinn_final.py
```

这会：
1. 加载 `epochs_best_model` 权重
2. 在 401×401 网格上推理
3. 读取 FEM 参考数据 `p_FBNS.txt` / `g_FBNS.txt`
4. 计算全部误差指标（P_rel_L2, MAE, IoU 等）
5. 生成 8 张对比图到 `comparison_results/`

---

## 预期训练结果

训练良好的话，`comparison_results/metrics.txt` 应该显示类似：

```
P_rel_L2                    = 1.39e-02    (1.4% 相对误差)
P_rel_Linf                  = 4.xxe-02
G_rel_L2                    = 1.18e-01
P_MAE                       = 3.84e-04
P_RMSE                      = 5.xxe-04
complementarity_violation   = 3.20e-03
cavitation_IoU              = 2.71e-01
cavitation_Dice             = 4.xxe-01
```

---

## 常见问题

### Q1: `tf.gradients` 报错 "module 'tensorflow' has no attribute 'gradients'"

TF 2.14+ 可能移除了 `tf.gradients`。修改 `reynold_pinn.py` 中的调用：

```python
# 将文件中所有的（约 9 处）
tf.gradients(xxx, yyy)
# 改为
tf.compat.v1.gradients(xxx, yyy)
```

或者在文件开头 import 之后加一行：
```python
import tensorflow as tf
tf.gradients = tf.compat.v1.gradients   # 兼容 TF 2.14+
```

### Q2: GPU 内存不足 (OOM)

减小 `Config` 中的 `N_f`（如改为 3000）。

```bash
# 也可以限制 TF 使用的 GPU 显存比例
export TF_FORCE_GPU_ALLOW_GROWTH=true
python reynold_pinn.py
```

### Q3: tensorflow-probability 安装报版本冲突

手动匹配版本（见步骤 4 的版本对应表）。如果 pip 仍报错，可以用 conda 装：

```bash
conda install -c conda-forge tensorflow-probability
```

### Q4: L-BFGS 报错

当前 `reynold_pinn.py` 的 `train_model` 函数中 `newton_iter=0`，已经跳过 L-BFGS，只用 Adam 训练。不影响结果。

### Q5: conda 不可用

```bash
# Ubuntu 上安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 重启终端后继续
```

### Q6: `libcudnn.so` 找不到

```bash
# 检查 cuDNN 是否正确安装
ldconfig -p | grep cudnn

# 如果没输出，手动添加到 LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 网络架构说明

训练使用的模型是 `new_neural_period_polar_exactBC_two_output` (switch=8)：

```
输入 (R, θ)
  │
  ▼
Coslayer_normalization     ← Fourier 特征编码
  cos(πθ+φ)·w_θ + R·w_R + bias → tanh            (512 参数)
  │
  ├─→ x_U = Dense(128, tanh)    ← U 分支
  ├─→ x_V = Dense(128, tanh)    ← V 分支
  ├─→ x_1 = Dense(1, tanh)      ← Hermite 插值参数
  ├─→ x_2 = Dense(1, tanh)      ← Hermite 插值参数
  │
  ▼
Gate 隐藏层 (4 层 × 128 宽，无残差连接)
  gate = tanh(Dense(128)(x))      ← 从当前 x 读取
  x = gate·x_U + (1-gate)·x_V     ← 门控混合
  │
  ▼
├─→ P:  Dense(1, glorot_normal)         → g_func + σ₁·P_raw → tanh²
└─→ γ:  Dense(1, kernel=1e-6, bias=0)   → σ₂·γ_raw → tanh²
```

**关键设计**：
- **Coslayer**：单层线性混合 + tanh，极简输入编码
- **Gate 门控**：`gate·U + (1-gate)·V`，无残差连接
- **PCGrad**：每个 loss term 独立求梯度后投影掉冲突分量，防止量级碾压
- **自适应损失权重**：`ComputeSum_weight` 的 EMA 机制动态平衡各 loss
- **硬 BC**：`atanh` 插值 + `sigma_func` 距离函数，边界精确满足
- **非负输出**：`tanh²` 保证 p ≥ 0, γ ≥ 0
