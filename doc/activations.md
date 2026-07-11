# 激活函数全景分析

> 本文逐层追踪网络中每一处激活函数，解释其选择原因和物理/数值含义。

---

## 目录

1. [总览：激活函数分布地图](#1-总览激活函数分布地图)
2. [输入层：Coslayer 的 cos + tanh](#2-输入层coslayer-的-cos--tanh)
3. [特征分支：U/V/Hermite 的 tanh](#3-特征分支uvhermite-的-tanh)
4. [门控机制：gate 的 tanh](#4-门控机制gate-的-tanh)
5. [残差连接的"无激活"传递](#5-残差连接的无激活传递)
6. [输出头：深层的 tanh + 残差](#6-输出头深层的-tanh--残差)
7. [BC 强制执行：exp + atanh](#7-bc-强制执行exp--atanh)
8. [最终输出：tanh² 的妙用](#8-最终输出tanh²-的妙用)
9. [PIKAN 专属：silu + B-spline 可学习激活函数](#9-pikan-专属silu--b-spline-可学习激活函数)
10. [汇总表](#10-汇总表)

---

## 1. 总览：激活函数分布地图

代码位置：[`torch_pinn/networks.py:331-424`](../torch_pinn/networks.py#L331-L424)

```
输入 (R, θ)
  │
  ├─[1] Coslayer ─────────────────────────────────────────────
  │     ├─ cos(π·θ_norm + φ)          ← Fourier 周期编码
  │     └─ tanh(Linear(concat(R_norm, cos_features)))
  │                                    ← 非线性 + 值域约束
  │
  │  输出: x [N, cos_units]    inputs_R [N, 1]
  │
  ├─[2] 特征分支 ─────────────────────────────────────────────
  │     ├─ U_base = tanh(Linear(x))    ← 特征通道 A
  │     ├─ V_base = tanh(Linear(x))    ← 特征通道 B (双通道解耦)
  │     ├─ x_1    = tanh(Linear(x))    ← Hermite 端点导数 1
  │     └─ x_2    = tanh(Linear(x))    ← Hermite 端点导数 2
  │
  ├─[3] 门控隐藏层 (×N_layers) ────────────────────────────────
  │     ├─ gate = tanh(Linear(x))      ← 门控信号 ∈ [-1,1]
  │     ├─ U_w  = tanh(Proj(U_base))   ← 宽度改变的 U
  │     ├─ V_w  = tanh(Proj(V_base))   ← 宽度改变的 V
  │     ├─ main = gate·U + (1-gate)·V  ← 门控融合 (无单独激活)
  │     └─ main += skip_proj(prev)     ← 残差连接 (无激活)
  │
  ├─[4] 输出头 ────────────────────────────────────────────────
  │     Pressure:  tanh(FC) → tanh(FC) [+ skip] → Linear → p_raw
  │     Gamma:     tanh(FC) → tanh(FC) [+ skip]
  │                 → concat(p_h2, g_h2)
  │                 → tanh(FC) → tanh(FC) [+ skip] → Linear → g_raw
  │
  ├─[5] BC 强制执行 ──────────────────────────────────────────
  │     ├─ atanh(线性插值(√P_o, √P_i))  ← g_func_1 (BC 解析)
  │     ├─ Hermite(x_1, x_2)             ← g_func_2 (BC 可学)
  │     └─ exp(边界距离)                  ← sigma_func (BC 距离)
  │
  └─[6] 最终输出 ─────────────────────────────────────────────
        ├─ P = tanh(g_func + sigma × p_raw)²    ← 非负压力
        └─ γ = tanh(sigma × g_raw)²             ← 非负空化
```

**全网络使用 3 种核心数学函数：tanh、cos、exp，加上输出层的平方。**

---

## 2. 输入层：Coslayer 的 cos + tanh

代码位置：[`torch_pinn/networks.py:61-100`](../torch_pinn/networks.py#L61-L100)

### 2.1 计算流程

```python
# Step 1: 归一化 R 和 θ 到 [-1, 1]
inputs_R     = 2 × (R - R_min)/(R_max - R_min) - 1     # [N, 1]
inputs_Theta = 2 × (θ - θ_min)/(θ_max - θ_min) - 1     # [N, 1]

# Step 2: Fourier 特征 = cos(π · θ_norm + φ)
outputs = cos(inputs_Theta × π + φ)                     # [N, units]

# Step 3: 组合 R_norm 和 cos 特征
outputs = kernel[0] × inputs_R + kernel[1] × cos_features + bias

# Step 4: 激活
outputs = tanh(outputs)
```

### 2.2 为什么用 cos？

| 理由 | 说明 |
|------|------|
| **天然周期编码** | cos(θ + 2π/6) = cos(θ)，自动处理周向 K=6 的周期性 |
| **傅里叶特征网络** | 和 Neural Tangent Kernel 理论一致：cos 特征使网络能学习高频函数 |
| **可训练频率** | K=π（固定），但相位 φ 是可训练的 → 网络学习最佳相位偏移 |
| **多频率隐式编码** | 多个 cos 单元 (`units=128`) 通过不同的 φ 值覆盖不同的频率成分 |

**数学本质**：`kernel[1] × cos(π·θ + φ)` 等价于 `A·cos(π·θ) + B·sin(π·θ)` 的傅里叶级数分解（通过不同 φ 实现）。

### 2.3 为什么 cos 输出后还要 tanh？

```
cos_features ∈ [-1, 1]  — 已经是有界的
     ↓ + R_norm 加权 + bias
combined    ∈ [-∞, +∞]  — 线性组合无界
     ↓ tanh
x           ∈ [-1, +1]  — 重新约束到有界范围
```

**原因**：
1. **值域归一化**：R_norm 和 cos_features 的线性组合可能超出 [-1, 1]，tanh 将其拉回
2. **非线性引入**：虽然 cos 已经是非线性的，但 `tanh(w₁·R + w₂·cos(θ+φ) + b)` 提供了 R 和 θ 之间的**交叉非线性交互**
3. **梯度友好**：tanh 的导数 `1-tanh²(x)` 在 [-1, 1] 区间平滑不为零

---

## 3. 特征分支：U/V/Hermite 的 tanh

代码位置：[`torch_pinn/networks.py:337-342`](../torch_pinn/networks.py#L337-L342)

```python
U_base = tanh(self.x_U(x))    # [N, base_width]  全连接 → tanh
V_base = tanh(self.x_V(x))    # [N, base_width]  全连接 → tanh
x_1    = tanh(self.x_1(x))    # [N, 1]           全连接 → tanh
x_2    = tanh(self.x_2(x))    # [N, 1]           全连接 → tanh
```

### 3.1 U/V 双分支的设计

这是该架构最特殊的设计——标准的 MLP 只有一条 `x → FC → activation → FC → ...` 路径，这里用了**并行双通道**：

```
  coslayer 输出 x ──┬── Linear_U ── tanh ── U_base  (通道 A)
                    ├── Linear_V ── tanh ── V_base  (通道 B)
                    ├── Linear_1 ── tanh ── x_1     (Hermite 端点 1)
                    └── Linear_2 ── tanh ── x_2     (Hermite 端点 2)
```

### 3.2 为什么用 tanh？

| 特性 | 说明 |
|------|------|
| **对称零中心** | tanh(x) ∈ [-1, 1]，均值为 0 → 梯度不偏向正或负 |
| **光滑饱和** | 在 ±1 附近梯度趋于 0，自然提供"软阈值" |
| **与门控配合** | gate ∈ [-1,1]（也是 tanh），U/V ∈ [-1,1]，三者值域一致 → 融合稳定 |
| **二阶导数连续** | tanh 无限光滑（C∞），PINN 需要二阶 autograd 导数→tanh 不会引入不光滑点 |

### 3.3 为什么不是 ReLU？

ReLU 在 PINN 中有一个致命问题：**二阶导数为零**。

```
ReLU(x)  = max(0, x)
ReLU'(x) = 0 (x<0) 或 1 (x>0)
ReLU''(x)= 0 (处处，除了不可导的 x=0)
```

雷诺方程需要 `∂²p/∂R²` 和 `∂²p/∂θ²`，如果所有中间激活都是 ReLU，二阶导数恒为零——物理完全无法学习。tanh 的二阶导数是 `-2·tanh(x)·(1-tanh²(x))`，处处非零。

---

## 4. 门控机制：gate 的 tanh

代码位置：[`torch_pinn/networks.py:349-361`](../torch_pinn/networks.py#L349-L361)

```python
for i, (w, gate_layer) in enumerate(zip(hidden_w, self.gate_layers)):
    gate = tanh(gate_layer(x))         # ← tanh 产生门控信号
    # ...
    main = gate * U_w + (1 - gate) * V_w
```

### 4.1 门控的物理直觉

```
gate → +1  →  main ≈ U_w   (纯通道 A 激活)
gate → -1  →  main ≈ 2*V_w (纯通道 B 激活 + 额外 V，因为 1-(-1)=2)
gate →  0  →  main ≈ 0.5×U_w + 0.5×V_w  (均等混合)
```

实际上 1-gate 的范围是 [0, 2]（因为 gate ∈ [-1,1]），所以：
- gate=-1 时 V 通道权重为 2
- gate=+1 时 V 通道权重为 0

### 4.2 为什么门控信号来自 x（Fourier 特征）而非 prev（上一隐藏层）？

| 方案 | 优劣 |
|------|------|
| `gate = tanh(Linear(prev))` | 每层门控只看到上一层的输出，丢失全局坐标信息 |
| `gate = tanh(Linear(x))` ✅ | **每层门控直接连接到 Fourier 特征 x**，保留了对原始坐标 (R,θ) 的感知 |

这是关键设计：**门控信号是坐标的函数**。同一个 (R,θ) 在所有隐藏层中通过 gate 决定用 U 还是 V 的激活模式。这让网络可以根据空间位置切换"行为模式"——例如在沟槽区 vs 台地区使用不同的特征表达。

### 4.3 为什么 gate 用 tanh？

- tanh ∈ [-1, 1] 天然适合双极性门控
- sigmoid ∈ [0, 1] 也可以做门控，但会导致 `1-gate ∈ [0, 1]`——V 通道永远不可能被放大
- tanh 允许 gate=-1 时 `1-gate=2`，让 V 通道在需要时可以产生 2× 的权重

---

## 5. 残差连接的"无激活"传递

代码位置：[`torch_pinn/networks.py:363-368`](../torch_pinn/networks.py#L363-L368)

```python
main = gate * U_w + (1.0 - gate) * V_w     # 门控融合

if prev is not None:
    skip = self.skip_proj[i - 1](prev)      # Linear 投影 (无激活)
    main = main + skip                       # 残差相加 (无激活)
```

### 5.1 skip_proj 为什么不用激活？

残差连接的标准做法（He et al., 2016）：

```
原始残差块:  x → Conv → BN → ReLU → Conv → BN → + → ReLU → 下一层
                                          ↑           ↑
                                       skip(x)     激活在加法之后

我们的残差:  prev → Linear(skip_proj) → + → 下一层 (tanh 在门控融合中已存在)
                                          ↑
                                      无额外激活
```

**原因**：
1. **Identity mapping 纯净性**：残差连接的本质是让梯度无障碍流通 `∂(main+skip)/∂prev = I + ∂main/∂prev`，如果 skip 上有激活函数，梯度被截断
2. **投影层的角色**：skip_proj 只是一个维度对齐的线性变换（或 identity），不需要激活
3. **Next-layer 会激活**：残差加了之后，`prev = main`，下一层迭代中的门控 U_w/V_w 会经过 tanh 激活

---

## 6. 输出头：深层的 tanh + 残差

代码位置：[`torch_pinn/networks.py:377-393`](../torch_pinn/networks.py#L377-L393)

### 6.1 Pressure Head

```python
p_h1  = tanh(self.p_fc1(prev))          # [N, H]
p_h2  = tanh(self.p_fc2(p_h1)) + p_h1   # [N, H]   ← tanh 残差块
p_raw = self.p_fc_out(p_h2)             # [N, 1]    ← 无激活！
```

### 6.2 Gamma Head (含 cross-talk)

```python
# Stage 1: 独立处理
g_h1  = tanh(self.g_fc1(prev))          # [N, H]
g_h2  = tanh(self.g_fc2(g_h1)) + g_h1   # [N, H]   ← tanh 残差块

# Stage 2: 融合压力信息
g_cat  = concat(p_h2, g_h2)             # [N, 2H]  ← cross-talk!
g_cat1 = tanh(self.g_cat_fc1(g_cat))    # [N, H]
g_cat2 = tanh(self.g_cat_fc2(g_cat1)) + g_cat1  # [N, H]  ← tanh 残差块
g_raw  = self.g_fc_out(g_cat2)          # [N, 1]   ← 无激活！
```

### 6.3 关键问题解答

**Q: 为什么输出头的各层用 tanh？**

输出头是"深度头"（2~3 层 FC + 残差），需要非线性来学习复杂的 P → γ 映射关系。tanh 提供光滑的非线性，同时保持二阶导数连续（PINN 需要）。

**Q: 为什么最后的 p_fc_out / g_fc_out 不用激活？**

因为 `p_raw` 和 `g_raw` 需要保持**无界的线性输出**——后续会被 `sigma_func`（∈ [0, p₃]）和 `tanh()` 缩放。如果在 FC_out 后加 tanh，值的范围被限制在 [-1, 1]，乘以 sigma 后范围更窄，会限制最终输出 P, γ ∈ [0, 1] 的动态范围。

```
有激活:  p_raw = tanh(Linear(p_h2)) ∈ [-1, 1]
         predictions = tanh(g_func + sigma × p_raw)² ∈ [tanh(BC±sigma)²]

无激活:  p_raw = Linear(p_h2) ∈ [-∞, +∞]           ← 更多自由度
         predictions = tanh(g_func + sigma × p_raw)² ∈ [0, 1]   ← tanh 在最后压缩
```

**Q: 残差块中的 `+ p_h1` 为什么在 tanh 之后？**

这是 **post-activation residual**（和 ResNet 的 pre-activation 相反）。好处是：
- `tanh(p_fc2(p_h1)) + p_h1` 保留了 p_h1 的线性分量（梯度高速公路）
- p_h1 ∈ [-1, 1]（tanh 输出），残差也在 [-1, 1] → 和的范围是 [-2, 2]，仍然有界

---

## 7. BC 强制执行：exp + atanh

代码位置：见 [`Out_Imp_BC_layer`](../torch_pinn/networks.py#L106-L133) 和 [`Out_Imp_BC_value_layer`](../torch_pinn/networks.py#L139-L177) 以及 forward 中的 `g_func_1` 计算。

### 7.1 sigma_func：指数边界距离（含 exp）

```python
sigma_func = p₃ × (1 - exp(p₁ × (-1 - R_norm))) × (1 - exp(p₂ × (R_norm - 1)))
```

| 位置 | R_norm | `1-exp(p₁(-1-R))` | `1-exp(p₂(R-1))` | sigma | 含义 |
|------|--------|--------------------|--------------------|-------|------|
| 内径 | -1 | 0 | ≈1 (exp 的大负值 → 0) | **0** | BC 硬件执行 |
| 外径 | +1 | ≈1 | 0 | **0** | BC 硬件执行 |
| 中心 | 0 | ≈1-exp(-p₁) | ≈1-exp(-p₂) | >0 | 网络可自由发挥 |

**为什么用 exp？** — exp 提供光滑但**快速**的边界衰减。p₁, p₂ 可学习，让网络自己决定边界过渡的"陡峭程度"。指数函数保证 sigma ≥ 0，且在边界处精确为零（不是近似！）。

### 7.2 g_func_1：atanh 变换

```python
g_func_1 = atanh( √P_o 和 √P_i 的线性插值 )
```

**为什么有 atanh？** 因为最终输出是 `P = tanh(g_func + sigma × p_raw)²`，取平方根后：
```
√P = tanh(g_func + sigma × p_raw)
```

要让边界处 P = P_bc，需要：
```
tanh(g_func_1|_boundary) = √P_bc
g_func_1|_boundary = atanh(√P_bc)
```

这是一个**解析逆变换**——用 atanh 把 BC 值"预编码"到 tanh 的定义域内，使得最终 tanh² 输出精确等于 BC 值。不是近似，是数学恒等。

### 7.3 g_func_2：Hermite 插值（纯多项式，无额外激活）

```python
g_func_2 = x_1 × (R+1) × ((R-1)/(-2))² + x_2 × (R-1) × ((R+1)/2)²
```

- Hermite 基函数是 R 的多项式（立方），不需要激活函数
- x_1 和 x_2 被 tanh 约束在 [-1, 1]，作为可学习的端点导数值
- 在边界处 `g_func_2` 精确为零（立方 Hermite 的性质）

---

## 8. 最终输出：tanh² 的妙用

代码位置：[`torch_pinn/networks.py:420-422`](../torch_pinn/networks.py#L420-L422)

```python
predictions  = tanh(g_func + sigma_func_1 × p_raw) ** 2    # 压力 P
prediction_g = tanh(sigma_func_2 × g_raw) ** 2              # 空化 γ
```

### 8.1 函数形状

```
f(x) = tanh(x)²

    1.0 ┤                ╭──────
        │              ╱╱
        │            ╱╱
    0.5 ┤          ╱╱
        │        ╱╱
        │      ╱╱
    0.0 ┤────╱╱────────────────
       -3   -1   0   1   3
```

| 性质 | 值 | 意义 |
|------|-----|------|
| **值域** | [0, 1] | 天然非负 |
| 在 x=0 处 | tanh(0)² = 0 | 零输入 → 零输出 |
| 导数 f'(x) | 2·tanh(x)·sech²(x) | x→0 时 ≈ 2x，线性响应 |
| 二阶导数 | 连续（C∞） | PINN 可以安全求导 |

### 8.2 为什么是 tanh² 而不是 softplus 或 ReLU？

| 候选 | 值域 | 问题 |
|------|------|------|
| `ReLU(x)` | [0, +∞) | 无上界 → P 可能爆炸到任意大 |
| `softplus(x)` | (0, +∞) | 同上，且二阶导数不平滑 |
| `sigmoid(x)` | (0, 1) | 有上界，但 x=0 时输出 0.5（零偏移） |
| `tanh(x)²` ✅ | [0, 1] | **有上下界，x=0→0，光滑，二阶导数连续** |

### 8.3 对空化模型的意义

物理上：
- **P ∈ [0, 1]**：P=0 代表空化压力，P>0 代表全膜压力
- **γ ∈ [0, 1]**：γ=0 代表无空化，γ>0 代表空化分数

`tanh²` 完美匹配这个物理约束：值域 [0,1]，在 0 处光滑过渡，在 x→∞ 时饱和于 1。

### 8.4 对 L_FB 小的直接贡献

因为 `tanh²` 天然非负，JFO 条件 `p+γ-√(p²+γ²)=0` 的唯一可能违反是 p 和 γ 同时 > 0。而 `tanh²` 不强制 p·γ=0，但它保证了 p≥0, γ≥0，使得 FB 条件至少有一半自动满足。

---

## 9. PIKAN 专属：silu + B-spline 可学习激活函数

代码位置：[`torch_pinn/networks.py:430-547`](../torch_pinn/networks.py#L430-L547)

### 9.1 KANLinear 的激活函数

```python
# KAN 层用两个并行路径替代 Linear + tanh:
y_j = Σ_i [ w_b_ij × silu(x_i) + w_s_ij × Σ_k c_ijk × B_k(x_i) ]
       └─── base path ───┘   └─────── spline path ─────────┘
```

### 9.2 SiLU (Sigmoid Linear Unit) — Base Path

```
silu(x) = x × sigmoid(x)
         = x / (1 + exp(-x))
```

```
        │         ╱
        │       ╱╱
    0   │─────╱╱──────────
        │   ╱╱
   -2   │ ╱╱
        └──────────────────
       -2   0   2
```

| 性质 | vs ReLU | vs tanh |
|------|---------|---------|
| 值域 | (-0.28, +∞) | (-1, 1) |
| 光滑性 | C∞（处处可导） | C∞ |
| 非单调 | 在 x<0 有微小负值 | 严格单调 |
| 二阶导 | 非零（关键！） | 非零 |
| 梯度消失 | 右侧无界增长 | 双侧饱和 |

**为什么 PIKAN 中 base path 用 silu 而不用 tanh？**

1. **B-spline path 已提供了光滑的非线性**，base path 的角色是提供**线性+小幅非线性**的基线
2. silu 在 x>0 时 ≈ ReLU（保留大梯度），在 x<0 时有小负值（提供负调节能力）
3. silu 是 KAN 论文的标准选择（arXiv:2403.07288）

### 9.3 B-Spline 基函数 — Spline Path

```
B_{i,0}(x) = 1   if t_i ≤ x < t_{i+1}   else 0       ← 阶 0：分段常数

B_{i,k}(x) = (x-t_i)/(t_{i+k}-t_i) × B_{i,k-1}(x)     ← Cox-de Boor 递推
           + (t_{i+k+1}-x)/(t_{i+k+1}-t_{i+1}) × B_{i+1,k-1}(x)
```

**这本质上是一组可学习的激活函数！**

- 标准 MLP：每个神经元的激活函数是**固定的**（tanh、ReLU...）
- PIKAN：每个**连接**（edge）有一个**独特的**激活函数 = Σ c_ijk × B_k(x)

```
网格点 G=5, 阶数 K=3:
────┼────┼────┼────┼────┼────
t0   t1   t2   t3   t4   t5   t6   t7   t8
└─────── G+2K+1 = 12 个节点 ──────┘
每个边有 G+K = 8 个 B-spline 系数

对每个 (out_j, in_i) 边:
  激活函数(x) = c_ij0×B_0(x) + c_ij1×B_1(x) + ... + c_ij7×B_7(x)
  这是该边上唯一的、可学习的激活形状
```

### 9.4 MLP vs PIKAN 的激活函数对比

| 维度 | MLP | PIKAN |
|------|-----|-------|
| 激活位置 | 神经元输出 | 连接边 |
| 激活形状 | 固定函数 (tanh) | 可学习的 B-spline 线性组合 |
| 训练参数 | 权重 + 偏置 | 权重 + 偏置 + spline 系数 |
| 表达能力 | 宽度补偿深度 | 激活函数的自适应形状补偿 |
| 参数量 | 更少（同结构） | 更多（每边额外 G+K 个系数） |

---

## 10. 汇总表

| # | 位置 | 代码行 | 函数 | 输入来源 | 输出去向 | 为何选它 |
|---|------|--------|------|----------|----------|----------|
| 1 | Coslayer 编码 | L82 | **cos** | θ_norm | 傅里叶特征 | 周期编码 + 高频能力 |
| 2 | Coslayer 激活 | L97-98 | **tanh** | 线性组合 | 特征 x | 值域约束 + 非线性 |
| 3 | U_base 分支 | L337 | **tanh** | Linear(x) | 门控融合 | 零中心 + 二阶导非零 |
| 4 | V_base 分支 | L338 | **tanh** | Linear(x) | 门控融合 | 零中心 + 二阶导非零 |
| 5 | Hermite x₁ | L341 | **tanh** | Linear(x) | g_func_2 | 端点导数值域约束 |
| 6 | Hermite x₂ | L342 | **tanh** | Linear(x) | g_func_2 | 端点导数值域约束 |
| 7 | Gate 信号 | L351 | **tanh** | Linear(x) | 门控权重 | 双极性 [-1,1] 适合门控 |
| 8 | U_proj 投影 | L357 | **tanh** | Linear(U_base) | 门控融合 | 统一值域 |
| 9 | V_proj 投影 | L358 | **tanh** | Linear(V_base) | 门控融合 | 统一值域 |
| 10 | **门控融合** | L361 | **无** | gate·U+(1-gate)·V | 隐藏状态 | 线性混合，激活在上游 |
| 11 | **残差相加** | L366 | **无** | main + skip | next prev | 梯度高速公路 |
| 12 | p 头 layer1 | L379 | **tanh** | Linear(prev) | p 头 layer2 | 非线性 |
| 13 | p 头 layer2 | L380 | **tanh+skip** | Linear(p_h1) + p_h1 | p_fc_out | 残差块的 tanh |
| 14 | p 头输出 | L381 | **无** | Linear(p_h2) | p_raw | 保持无界线性范围 |
| 15 | g 头 layer1 | L385 | **tanh** | Linear(prev) | g 头 layer2 | 非线性 |
| 16 | g 头 layer2 | L386 | **tanh+skip** | Linear(g_h1) + g_h1 | concat | 残差块的 tanh |
| 17 | g 头 cross-talk1 | L391 | **tanh** | Linear(concat) | g cross-talk2 | cross-talk 非线性 |
| 18 | g 头 cross-talk2 | L392 | **tanh+skip** | Linear(g_cat1) + g_cat1 | g_fc_out | 残差块的 tanh |
| 19 | g 头输出 | L393 | **无** | Linear(g_cat2) | g_raw | 保持无界线性范围 |
| 20 | BC sigma | L131 | **exp** | 可学参数 | sigma_func | 光滑边界衰减 |
| 21 | BC g_func_1 | L403 | **atanh** | √BC 插值 | g_func | **解析逆变换**保证 BC 精确 |
| 22 | 最终 P 输出 | L421 | **tanh²** | g_func+sigma×p_raw | **最终 P** | 非负 [0,1] + C∞ 光滑 |
| 23 | 最终 γ 输出 | L422 | **tanh²** | sigma×g_raw | **最终 γ** | 非负 [0,1] + C∞ 光滑 |
| — | — | — | — | — | — | — |
| 24 | PIKAN base | L540 | **silu** | x | 线性组合 | KAN 标准选择 |
| 25 | PIKAN spline | L486-530 | **B-spline** | x | 线性组合 | 可学习激活函数形状 |

### 各函数出现次数统计（MLP 版本）

```
tanh:  14 处 (网络中最核心的激活函数)
无激活:  5 处 (Linear 输出层 + 残差 + 门控融合)
exp:    1 处 (BC sigma 函数)
atanh:  1 处 (BC 逆变换)
cos:    1 处 (Fourier 编码)
tanh²:  2 处 (最终 P 和 γ 输出)
```

**结论：tanh 是这座网络的主角。** 选择 tanh 的根本原因是 PINN 对二阶导数的要求——只有光滑的激活函数才能让 autograd 产生有意义的 `∂²p/∂R²` 和 `∂²p/∂θ²`。

---

*文档生成日期：2026-07-11*
