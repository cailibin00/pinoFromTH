# 0504 原版 TF PINN 运行指南（Windows — 本机专用）

> 以下配置针对这台电脑实测编写
> - **GPU**：NVIDIA GeForce RTX 4060 Laptop（8 GB VRAM）
> - **驱动**：566.14 · **CUDA Driver**：12.7 · **nvcc**：12.4（不重要，conda 自带工具链）
> - **Conda**：24.11.3（位于 `D:\Anaconda`）
> - **项目路径**：`E:\a_lab\lab_Pro\pinoFromTH\0504`

---

## 为什么需要新建 Python 3.10 环境

| 当前 | 需要 |
|------|------|
| base Python **3.12** | **3.10** |
| TF 不支持 Python 3.12 | TF 2.10 最高支持 3.10 |

**关键选择：TF 2.10.0** —— 最后一个原生支持 Windows GPU 的 TensorFlow 版本。TF 2.11 起 Windows GPU 需要 WSL2。

---

## 安装步骤（每个命令直接复制执行）

### 步骤 1：创建 Python 3.10 环境

打开 **PowerShell** 或 **Anaconda Prompt**：

```powershell
conda create -n tf0504 python=3.10 -y
conda activate tf0504
```

### 步骤 2：安装 CUDA Toolkit + cuDNN（conda 一键搞定）

```powershell
# conda 会下载 CUDA 11.2 + cuDNN 8.1，与系统已装的 CUDA 12.4 不冲突
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
```

> 驱动 566.14（CUDA 12.7）**向下兼容** CUDA 11.2 toolkit，不用担心。

### 步骤 3：安装 TensorFlow 2.10 + 配套包

```powershell
pip install tensorflow==2.10.0
pip install tensorflow-probability==0.18.0
pip install numpy scipy matplotlib tqdm pyDOE2 pyfiglet
```

### 步骤 4：验证

```powershell
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

预期输出：
```
TF: 2.10.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

如果 GPU 列表为空，执行：
```powershell
mkdir -p $env:APPDATA\pip
echo "[global]" > $env:APPDATA\pip\pip.ini
echo "extra-index-url = https://pypi.nvidia.com" >> $env:APPDATA\pip\pip.ini
pip install --upgrade tensorflow==2.10.0
```

---

## 运行

### 训练

```powershell
conda activate tf0504
cd E:\a_lab\lab_Pro\pinoFromTH\0504
python reynold_pinn.py
```

训练参数速览（`Config` 类中可改）：

| 参数 | 值 | 含义 |
|------|-----|------|
| N_f | 4900 | 配点数 |
| N_train | 5000 | 每轮迭代 |
| NL_train | 4 | RAD 轮数 |
| layer_sizes | [2,128,128,128,128,2] | 网络 |
| **总迭代** | **80000** | 4 阶段 × 4 轮 × 5000 |

训练完自动生成：
- `epochs_best_model.data-00000-of-00001` / `.index` — 最佳权重
- `reynolds_pinn_N4900_iter80000/` — SavedModel
- `reynolds_pinn_N4900_iter80000_pressure_contour.png` 等 4 张图

### 仅评估（用已有权重）

```powershell
conda activate tf0504
cd E:\a_lab\lab_Pro\pinoFromTH\0504
python compare_fem_pinn_final.py
```

输出到 `comparison_results/`，包括 `metrics.txt` + 8 张对比图。

---

## 常见问题

### Q1: CUDA/cuDNN 装不上

网络问题换清华源：
```powershell
conda install -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge cudatoolkit=11.2 cudnn=8.1 -y
```

### Q2: GPU 内存不足 (8 GB)

RTX 4060 的 8 GB 跑这个模型（58 万参数、4900 配点）绰绰有余。如果配点加多后 OOM：

```powershell
set TF_FORCE_GPU_ALLOW_GROWTH=true
python reynold_pinn.py
```

### Q3: 训练到一半崩溃

检查散热，笔记本 GPU 温度超过 86°C 会降频。建议用支架垫高或限制功耗：
```powershell
# 限制 GPU 功耗到 80W（4060 默认 108W）
nvidia-smi -pl 80
```

### Q4: 想用更新的 TF + GPU

TF 2.11+ 在 Windows 上 GPU 需要 **WSL2**。如果愿意折腾：
1. 装 WSL2 + Ubuntu 22.04
2. 用上一版文档的 Linux 教程
3. 路径从 `\\wsl$\Ubuntu\...` 或直接 `cd /mnt/e/a_lab/lab_Pro/pinoFromTH/0504` 访问

---

## 网络架构（回顾）

```
输入 (R, θ)
  │
  ▼ Coslayer: cos(πθ+φ)·w_θ + R·w_R + bias → tanh  (512参数)
  ├─ x_U, x_V = Dense(128, tanh)
  ├─ x_1, x_2 = Dense(1, tanh)
  │
  ▼ Gate×4: gate=tanh(Dense(x)), x=gate·U+(1-gate)·V
  │
  ▼ P: Dense(1) → g_func+σ₁·P → tanh²
  └ γ: Dense(1, w=1e-6) → σ₂·γ → tanh²
```

PCGrad + 自适应权重 + 硬 BC + tanh² 非负。
