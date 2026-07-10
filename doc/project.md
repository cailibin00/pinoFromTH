# Reynolds Equation PINN Solver — 螺旋槽止推轴承空化问题

> **Physics-Informed Neural Network (PINN)** 求解含 JFO 空化模型的雷诺方程。
> PyTorch 实现，忠实复刻 0504 TensorFlow 版本。

---

## 目录

1. [物理背景](#1-物理背景)
2. [数学模型](#2-数学模型)
3. [神经网络架构](#3-神经网络架构)
4. [损失函数与训练策略](#4-损失函数与训练策略)
5. [项目结构](#5-项目结构)
6. [快速开始](#6-快速开始)
7. [配置参数](#7-配置参数)
8. [FEM 对比验证](#8-fem-对比验证)

---

## 1. 物理背景

### 问题描述

螺旋槽止推轴承（Spiral-Groove Thrust Bearing）是一种流体动压轴承。旋转运动将润滑油从内径泵向外径，在收敛间隙中产生压力场，从而承受轴向载荷。当局部压力降至空化压力以下时，润滑油膜破裂产生空化（cavitation）——这是本问题的核心难点。

### 几何结构

```
         外径 r_o (52mm)
    ┌─────────────────────┐
    │   螺旋槽区域         │  ← 对数螺旋线, 螺旋角 α=3°
    │   / / / / / / / /   │     6 个周期性扇区 (K=6)
    │  / / / / / / / /    │
    │ ═══════════════════ │  ← 内径 r_i (47mm)
    └─────────────────────┘
         槽深 = 3×h_i (9μm)
         平衡膜厚 h_i = 3μm
```

| 参数 | 符号 | 值 | 单位 |
|------|------|-----|------|
| 内径 | r_i | 47.0 | mm |
| 外径 | r_o | 52.0 | mm |
| 平衡膜厚 | h_i | 3.0 | μm |
| 周期数 | K | 6 | — |
| 螺旋角 | α | 3 | ° |
| 槽深比 | h_groove / h_i | 3.0 | — |
| 槽宽比 | groove_ratio | 0.5 | — |
| 转速 | ω | 6000 | rpm |
| 动力粘度 | η | 8.0×10⁻⁴ | Pa·s |
| 内径压力 | p_i | 0.1 | MPa |
| 外径压力 | p_o | 0.15 | MPa |

### 膜厚分布 H(R,θ)

膜厚函数是分段常数（槽区 vs 非槽区），使用 **sigmoid 平滑过渡**：

$$H(R,\theta) = \begin{cases} 1 & \text{非槽区（平滑过渡）} \\ 1 + h_{texture}/h_i & \text{槽区（平滑过渡）} \end{cases}$$

- 槽区由对数螺旋线方程定义：$\theta_{sym}(R) = \frac{\ln(R/r_g)}{\tan\alpha} + \theta_{offset}$
- 径向过渡范围 (R_d_1, R_d_2)：sigmoid 平滑
- 周向过渡：sigmoid 平滑，周期扩展覆盖 5 个相邻扇区

---

## 2. 数学模型

### 2.1 无量纲化

长度尺度：$r_{base} = r_o$；压力尺度：$p_{base} = 10 \cdot p_o$

无量纲轴承数：$\Lambda = \frac{6\eta\omega r_{base}^2}{h_i^2 p_{base}}$

### 2.2 雷诺方程 (极坐标)

稳态不可压缩雷诺方程在极坐标 $(R, \theta)$ 下：

$$\frac{1}{R}\frac{\partial}{\partial R}\left(R H^3 \frac{\partial P}{\partial R}\right) + \frac{1}{R^2}\frac{\partial}{\partial\theta}\left(H^3 \frac{\partial P}{\partial\theta}\right) - \Lambda\frac{\partial H}{\partial\theta} + \Lambda\frac{\partial(\gamma H)}{\partial\theta} = 0$$

其中：
- $P(R,\theta)$：无量纲压力（非负）
- $\gamma(R,\theta)$：空化率（$\gamma \in [0,1]$，1=完全空化）
- $H(R,\theta)$：无量纲膜厚

### 2.3 JFO 空化模型 (Fischer-Burmeister 互补条件)

JFO 模型要求压力和空化率满足互补条件：**压力 > 0 处无空化，空化 > 0 处无压力**。

使用 **Fischer-Burmeister (FB) NCP 函数**将互补条件转化为等式约束：

$$P + \gamma - \sqrt{P^2 + \gamma^2} = 0$$

这等价于经典的 $P \geq 0, \gamma \geq 0, P \cdot \gamma = 0$ 互补三联条件。

### 2.4 JFO 稳定项

在雷诺方程中加入迎风型稳定项以处理空化边界的数值振荡：

$$\mathcal{R}_{stab} = \frac{\partial^2\gamma}{\partial\theta^2} \cdot \tau \cdot \tau_2$$

其中：
- $\tau = (|\partial\gamma/\partial\theta| - \partial\gamma/\partial\theta) \cdot \epsilon$（空化率梯度为正时激活）
- $\tau_2 = (\partial P/\partial\theta - |\partial P/\partial\theta|) \cdot \epsilon$（压力梯度为负时激活）
- $\epsilon = 0.1$（稳定强度）

### 2.5 边界条件

- **内径** (R = r_i/r_o)：$P = P_i$（固定入口压力，Dirichlet BC）
- **外径** (R = 1)：$P = P_o$（固定出口压力，Dirichlet BC）
- **周向**：周期性（由 `Coslayer_normalization` 层自动保证）

---

## 3. 神经网络架构

主模型 `new_neural_period_polar_exactBC_two_output`（u_model_switch=8）采用以下关键设计：

### 3.1 整体结构

```
输入: [R, θ]  (N×2)
  │
  ├─ Coslayer_normalization (Fourier 特征编码层)
  │   ├─ 归一化 R,θ → [-1, 1]
  │   ├─ cos(π·θ_norm + φ)  ← 可训练相位 φ
  │   ├─ kernel[0,:]·R_norm + kernel[1,:]·cos(…) + bias
  │   └─ tanh 激活 → (x, R_norm)
  │
  ├─ 门控分支
  │   ├─ x_U = Dense(128, tanh)(x)   ← U 分支
  │   ├─ x_V = Dense(128, tanh)(x)   ← V 分支
  │   ├─ x_1 = Dense(1, tanh)(x)     ← Hermite 插值分支1
  │   └─ x_2 = Dense(1, tanh)(x)     ← Hermite 插值分支2
  │
  ├─ 隐藏层（3 层，每层带门控）
  │   for each layer:
  │       x_t = Dense(128, tanh)(x)              ← 门控值
  │       x = x_t * x_U + (1 - x_t) * x_V         ← 门控融合
  │
  ├─ 双头输出
  │   ├─ p_raw  = Dense(1, linear)(x)
  │   └─ γ_raw  = Dense(1, linear)(x)
  │
  ├─ BC 强制执行（Hard Constraint）
  │   ├─ σ₁ = Out_Imp_BC_layer(R_norm)           ← 可学习 σ(R)
  │   ├─ σ₂ = Out_Imp_BC_layer(R_norm)           ← 可学习 σ(R)
  │   ├─ g₁ = atanh(√bc 线性插值)                 ← atanh 基函数
  │   ├─ g₂ = x₁·Hermite_base₁ + x₂·Hermite_base₂ ← Hermite 插值
  │   ├─ P = tanh²(g₁ + g₂ + σ₁·p_raw)           ← 压力输出 [0, P_max]
  │   └─ γ = tanh²(σ₂·γ_raw)                     ← 空化率输出 [0, 1]
  │
输出: [P, γ]  (N×1 each)
```

### 3.2 关键层详解

#### Coslayer_normalization — Fourier 特征编码

$$
\text{output}_i = \text{tanh}\left(W_{0,i} \cdot R_{norm} + W_{1,i} \cdot \cos(\pi \cdot \theta_{norm} + \phi_i) + b_i\right)
$$

- `kernel [2, units]`：可训练权重，学习 R 和 cos 特征的组合方式
- `phy [units]`：可训练相位，控制各频率分量的相位偏移
- K = π（固定），提供基础的傅里叶频率
- **作用**：将周期性周向坐标编码为正交傅里叶基，使网络天然满足 2π/K 周期性边界条件

#### Out_Imp_BC_layer — 可学习 BC 距离函数

$$\sigma(R_{norm}) = p_3 \cdot \big(1 - e^{p_1(-1-R_{norm})}\big) \cdot \big(1 - e^{p_2(R_{norm}-1)}\big)$$

- p₁, p₂, p₃ 可训练：控制 σ 在两边界处的衰减速率和整体幅值
- **性质**：σ(-1) = σ(+1) = 0，确保输出在 R 边界处精确满足 BC
- **优势**：相比固定多项式 σ = (1-R²)，可学习的指数形式能自适应调整形状

#### Out_Imp_BC_value_layer — Hermite 插值

$$g(R) = bc_0 \cdot H_{00}(R) \cdot H_{10}^2(R) + bc_1 \cdot H_{01}(R) \cdot H_{11}^2(R) + m_0 \cdot (R+1) \cdot H_{10}^2(R) + m_1 \cdot (R-1) \cdot H_{11}^2(R)$$

二点三次 Hermite 插值，m₀, m₁ 可训练：提供两边界处导数的自由度

#### 门控融合

$$x_{new} = \text{gate}(x) \cdot U(x) + (1 - \text{gate}(x)) \cdot V(x)$$

- U/V 两分支学习不同的特征表示
- 门控网络（每个隐藏层独立）学习逐元素地在 U 和 V 之间插值
- 等价于：每个神经元自适应地在两种特征处理模式之间切换

### 3.3 模型规模

| 配置 | layer_sizes | 参数量 |
|------|-------------|--------|
| 默认 | [2, 128, 128, 128, 128, 2] | ~83,600 |

---

## 4. 损失函数与训练策略

### 4.1 损失项

| 损失项 | 含义 | 公式 |
|--------|------|------|
| L_Reynolds | 雷诺方程残差 | MSE(f_FBNS, 0) |
| L_FB | FB 互补条件 | MSE(P+γ-√(P²+γ²), 0) |

### 4.2 PCGrad — 投影冲突梯度

多任务学习中，不同损失项的梯度可能相互冲突（方向夹角 > 90°）。PCGrad 算法：

1. 对各损失项分别计算梯度：$\{g_1, g_2, \ldots, g_n\}$
2. 随机打乱梯度顺序
3. 对每个 $g_i$：遍历所有 $g_j$，若 $\cos(g_i, g_j) < 0$（方向冲突），则从 $g_i$ 中减去 $g_j$ 上的投影分量
4. 合并所有投影后梯度：$g_{final} = \sum_i g_i^{proj}$

**效果**：消除任务间梯度冲突，防止某个任务主导训练方向

### 4.3 自适应损失权重

匹配 TF 的 `ComputeSum_weight` 层：

$$w_i^{new} = (1-\alpha) \cdot w_i^{old} + \alpha \cdot \frac{\overline{|\nabla_w L_{res}|}}{\overline{|\nabla_w L_i|}}$$

- $\alpha = 0.2$（平滑系数）
- 基于各损失项对网络权重的平均梯度幅值比，自动平衡各损失项的贡献
- 权重被截断至 $[10^{-2}, 10^{12}]$ 防止数值溢出

### 4.4 训练流程

```
4 阶段 × (4 轮 RAD 细化 × 5000 步 Adam)
│
├─ 阶段 1: lr = 1e-3 → 1e-4 → 1e-5 (分段常数衰减)
├─ 阶段 2: lr = 1e-4 → 1e-4 → 1e-5
├─ 阶段 3: lr = 1e-5 → 1e-4 → 1e-5
└─ 阶段 4: lr = 1e-5 → 1e-5 → 1e-6

每轮后: RAD 自适应配点细化
```

**总训练步数**：4 阶段 × 4 轮 × 5000 步 = **80,000 步**

### 4.5 RAD — 残差自适应分布

训练过程中动态调整配点分布，将更多点放置在 PDE 残差大的区域：

1. 在测试点上评估 PDE 残差 $r(x)$
2. 计算采样概率：$p(x) \propto r(x)^{2k} + c$
3. 按概率采样新配点，追加到训练集中

参数：$k=1, c=10^{-16}$，每轮添加 $\{3\%,\;1\%\}$ 的原始点数。

### 4.6 优化器

| 阶段 | 优化器 | 参数 |
|------|--------|------|
| 阶段 1-4 | Adam | lr 分段衰减, β₁=0.99, β₂=0.999 |
| （可选） | L-BFGS | history_size=50, lr=0.8, tol=1e-12 |

---

## 5. 项目结构

```
pinoFromTH/
├── 0504/                                    # TensorFlow 参考实现（只读）
│   ├── reynold_pinn.py                      # TF 训练脚本
│   ├── tensordiffeq/                        # TF PINN 框架
│   └── ...
│
├── torch_pinn/                              # ★ PyTorch PINN 框架（新建）
│   ├── networks.py                          # 神经网络架构
│   │   ├── Coslayer_normalization           # Fourier 特征层
│   │   ├── Out_Imp_BC_layer                 # 可学习 BC 距离函数
│   │   ├── Out_Imp_BC_value_layer           # Hermite 插值层
│   │   └── new_neural_period_polar_exactBC_two_output  # 主模型
│   ├── models.py                            # 核心求解器
│   │   ├── ComputeSum_weight                # 自适应损失权重
│   │   └── CollocationSolverND              # 完整 PINN 求解器
│   ├── fit.py                               # 训练循环 (Adam + L-BFGS)
│   ├── pcgrad.py                            # PCGrad 梯度投影
│   ├── optimizers.py                        # L-BFGS 优化器
│   ├── domains.py                           # 计算域 + 配点生成
│   ├── boundaries.py                        # 边界条件 (Dirichlet 等)
│   ├── utils.py                             # 工具函数
│   ├── sampling.py                          # LHS 采样 (ESE 算法)
│   └── __init__.py                          # 公共 API
│
├── reynold_pinn_torch.py                    # ★ 主训练脚本
├── compare_fem_pinn_final_torch.py           # ★ FEM vs PINN 对比 (8图)
├── compare_fem_pinn_iso_v16_torch.py         # ★ FEM vs PINN 对比 (9图+等值线)
│
├── p_FBNS.txt                               # FEM 参考压力数据
├── g_FBNS.txt                               # FEM 参考空化数据
├── project.md                               # ★ 本文档
│
└── output_torch/                            # 训练输出（自动创建）
    ├── checkpoints/                         # 模型检查点
    ├── models/                              # 完整模型保存
    ├── figures/                             # 训练曲线图
    ├── log/                                 # 训练日志
    └── comparison_results*/                 # FEM 对比结果
```

---

## 6. 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, SciPy, tqdm

### 训练模型

```bash
# 默认配置（自动检测 CUDA, 全批量训练）
python reynold_pinn_torch.py

# 自定义设备和批量大小
# 在 Config 类中修改:
#   device = "cuda"      # 或 "cpu" / "auto"
#   batch_size = 2048    # 或 None (全批量)
```

### FEM 对比验证

```bash
# 训练完成后，运行对比脚本
python compare_fem_pinn_final_torch.py        # 8 图对比
python compare_fem_pinn_iso_v16_torch.py      # 9 图 + 等值线分析
```

### 预期结果

参考 TF 版本结果（`output/comparison_results/metrics.txt`）：

| 指标 | 参考值 |
|------|--------|
| P_rel_L2 | ~1.4×10⁻² (1.4%) |
| G_rel_L2 | ~1.2×10⁻¹ (12%) |
| P_MAE | ~3.8×10⁻⁴ |
| Cavitation IoU | ~0.27 |
| Cavitation Dice | ~0.43 |
| FB violation | ~3.2×10⁻³ |

---

## 7. 配置参数

所有可调参数集中在 `reynold_pinn_torch.py → Config` 类中：

```python
class Config:
    # ── 几何参数 ──
    r_i = 47.0e-3           # 内径 (m)
    r_o = 52.0e-3           # 外径 (m)
    h_i = 3.0e-6            # 平衡膜厚 (m)
    K = 6.0                 # 周期数

    # ── 螺旋槽 ──
    alpha_deg = 3.0         # 螺旋角 (°)
    h_texture_ratio = 3.0   # 槽深比
    groove_ratio = 0.5      # 槽宽比

    # ── 工况 ──
    p_i = 0.1e6             # 内径压力 (Pa)
    p_o_ratio = 1.5         # 外径/内径压力比
    eta = 8.00e-4           # 动力粘度 (Pa·s)
    omega_rpm = 6000        # 转速 (rpm)

    # ── 数值 ──
    N_f = 4900              # 配点数
    domain_fidelity = 50    # 域网格密度

    # ── 训练 ──
    layer_sizes = [2, 128, 128, 128, 128, 2]  # 网络结构
    N_train = 5000          # 每阶段训练迭代数
    NL_train = 4            # RAD 细化轮数

    # ── 硬件 ──
    device = "auto"         # "cuda" / "cpu" / "auto"
    batch_size = None       # None=全批量 / int=小批量
```

---

## 8. FEM 对比验证

### 数据格式

FEM 数据文件 `p_FBNS.txt` / `g_FBNS.txt` 为三列格式：
```
R   theta   value
```
在 201×201 规则网格上包含 40,401 个数据点。

### 对比指标

| 类型 | 指标 | 说明 |
|------|------|------|
| 全局 | rel_L2, rel_Linf, MAE, RMSE | 压力和空化率的整体精度 |
| 分区 | cavRegion / fullRegion | 分别评估空化区和全膜区 |
| 互补 | P·g 最大值 | JFO 条件违反量（应→0） |
| 形状 | IoU, Dice | 空化区域形状匹配度 |
| 边界 | 等值线位置偏差 | 空化边界的空间精度 |

### 可视化输出

| 图号 | 内容 |
|------|------|
| fig1 | FEM vs PINN 场量对比 (P, g 云图) |
| fig2 | 逐点绝对误差热力图 |
| fig3 | 空化边界叠加 (g=1e-6 等值线) |
| fig4 | 压力等值线叠加 |
| fig5 | 沿中截面线图 (R-mid, θ-mid) |
| fig6 | 周期性边界检查 |
| fig7 | 散点相关图 + R² |
| fig8 | JFO 互补条件可视化 (P×g) |
| fig9 | 空化等值线位置分析 (仅 iso_v16 版本) |

---

## 参考文献

- Jakobsson, Floberg, Olsson — JFO cavitation model
- Fischer-Burmeister NCP function for complementarity
- Yu et al. — PCGrad: Projected Conflicting Gradients for multi-task learning
- Jin, Chen, Sudjianto (2005) — Enhanced Stochastic Evolutionary (ESE) algorithm for LHS
- Lu et al. — Fourier feature networks for PINN
