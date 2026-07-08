# 模型架构改进日志

> **日期**: 2026-06-05
> **基于**: 原始代码 v1.1 (叶萌)
> **原则**: 修改应**单一纯粹** — 输入解耦留住，其余一切从简

---

## 改进总览

| 问题 | 旧架构 (switch=8) | 新架构 (switch=13) |
|------|------------------|-------------------|
| 输入编码 | R: 单标量线性投影, θ: cos特征; 加法融合, 互相干扰 | R/θ: 独立 Fourier→MLP→concat, 信息解耦 |
| 边界约束 | `g_func_1` (atanh-线性) + `g_func_2` (2-自由度Hermite) + 指数σ | `atanh(线性插值√P)` + `σ=1−R²`, 纯数学公式, 无可训练权重 |
| PDE 残差 | 标准 Reynolds 方程 | 不变 (无 stop_gradient, 无课程权重, 无逐点降权) |
| 训练 | 4阶段 LR 衰减 + RAD 重采样 | 不变 (无 w_wedge 等额外调度) |

---

## 1. 输入编码：Fourier 解耦

### 问题

旧架构 `Coslayer_normalization`:
- R 坐标只做 `R × kernel_1[i]` → **单标量线性投影**, 无法表达 R 方向的高频变化
- θ 坐标做 `cos(θ·K + φ)` → 周期性好
- R 和 θ 通过**加法融合**: `R*kernel_1 + cos(θ)*kernel_2 + bias` — 互相污染

### 新架构

新增 `new_neural_fourier_decoupled()` ([networks.py:811](tensordiffeq/networks.py#L811)):

```
输入 [R, θ]
    ├── R → normalize [-1,1] → [sin(2ⁱπR), cos(2ⁱπR)]₀³ → MLP(32,64) → R_embed (64d)
    └── θ → normalize [-1,1] → [sin(2ⁱπθ), cos(2ⁱπθ)]₀³ → MLP(32,64) → θ_embed (64d)

    Concat → 128d → U/V branching → [NN_P, NN_γ]
```

- R 获得 4 频率 × 2 (sin+cos) = 8 个 Fourier 特征, 能表达高频变化
- θ 同样 8 个 Fourier 特征, 天然周期性
- concat 保持通道独立 — 网络可选择性忽略某个坐标

---

## 2. 边界条件：简洁硬约束

### 核心公式

```python
g(R) = atanh( 0.5·(1-R_norm)·√P_i  +  0.5·(1+R_norm)·√P_o )
σ(R) = 1 − R_norm²
P = tanh( g(R) + σ(R)·NN_P )²
```

### 验证

```
在 R_norm = −1 (R_min):  σ=0, g=atanh(√P_i), P=tanh(atanh(√P_i))²=(√P_i)²=P_i  ✓
在 R_norm = +1 (R_max):  σ=0, g=atanh(√P_o), P=tanh(atanh(√P_o))²=(√P_o)²=P_o  ✓
在内部任意点:           σ≠0, NN 可自由修正 g(R) 的偏差
```

### 与旧对比

| 组件 | 旧 | 新 |
|------|----|----|
| g(R) | `atanh(线性(√P_i,√P_o))` + `x₁(R+1)((R-1)/(-2))²+x₂(R-1)((R+1)/2)²` | `atanh(线性(√P_i,√P_o))` |
| 自由度 | 1+2=3 | 1 |
| 训练参数 | x₁,x₂ (2个) | 0个 |
| σ(R) | `p₃(1-e^{p₁(-1-R)})(1-e^{p₂(R-1)})` | `1−R²` |
| 训练参数 | p₁,p₂,p₃ (3个) | 0个 |

**为什么好**: g(R) 和 σ(R) 是纯数学的 — atanh 是 tanh² 的反函数预补偿, 1−R² 是最简单的边界归零函数。内部偏差全部交给 NN, 不在 BC 机制上浪费参数。

---

## 3. 自适应膜厚过渡宽度

放弃 hermite, 回归 sigmoid — 但把过渡宽度 ξ 做成可训练变量.

### 原理

```python
log_ξ_theta = tf.Variable(log(域宽/30))  # 初值 ≈ 3.3%
log_ξ_R     = tf.Variable(log(域宽/40))  # 初值 ≈ 2.5%

H 中的 sigmoid: σ((θ − θ_sym(R)) / exp(log_ξ_theta))
```

### 优化

每个 RAD 重采样后, 对 ξ 做 50 步梯度下降:

```python
loss_ξ = MSE(f_p, 0)  # 跟主网络共享同一个优化目标
```

优化器自动平衡:
- ξ 太小 → ∂H/∂θ 太陡 → f_p 爆炸 → 梯度推大 ξ
- ξ 太大 → 膜厚模糊 → H 偏离真值 → 也增大 f_p → 梯度推小 ξ

ξ 被剪切在 `[exp(-8), exp(-2)]` ≈ `[0.03%, 13%]` 域宽之间.

### 效果

- ξ 在每个 RAD 循环后自适应调整, 随着训练进度梯度变小 ξ 可逐渐收窄
- 不需要手动调宽度, 不需要课程学习

---

## 修改的文件

### `tensordiffeq/networks.py`
- 新增 `new_neural_fourier_decoupled()` — 约 115 行

### `tensordiffeq/models.py`
- `compile()` 新增 `num_freq=4, embed_dim=64` 参数
- 新增 `u_model_switch=13` 分支

### `reynold_pinn.py`
- Config: 新增 `u_model_switch=13`, `num_fourier_freq=4`, `embed_dim=64`
- `create_H_func`: 回归 sigmoid, 过渡宽度 ξ 做成可学习 `tf.Variable` (log 空间)
- `create_pde_models`: 标准版, 无 stop_gradient/point_weight
- `train_model` / `_optimize_xi_step`: 每 RAD 后对 ξ 做梯度下降
- `layer_sizes`: `[2, 128, 128, 128, 128, 128, 2]` (6层)

### `compare_fem_pinn_final.py` / `compare_fem_pinn_iso_v16.py`
- 同步更新 `create_H_func` 返回值解构 (2→3), `create_pde_models` (3→2)

---

## 使用

```bash
python reynold_pinn.py
```

默认使用新架构 (`u_model_switch=13`)。回退旧架构: 在 Config 中设 `u_model_switch=8`.
