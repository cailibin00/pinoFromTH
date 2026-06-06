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

## 3. PDE 残差与训练：不变

- PDE 残差公式完全不变: Poiseuille + 楔形 + τ稳定化
- 训练流程完全不变: 4阶段分段LR + PCGrad + RAD重采样
- 损失函数不变: `L_total = w₁·L_Reynolds + w₂·L_FB + w₃·L_BC`

**核心信念**: Fourier 输入 + 简洁硬 BC 已经足够好, 不需要额外的梯度操纵或课程学习。

---

## 修改的文件

### `tensordiffeq/networks.py`
- 新增 `new_neural_fourier_decoupled()` — 约 115 行
- 旧函数全部保留不动

### `tensordiffeq/models.py`
- `compile()` 新增 `num_freq=4, embed_dim=64` 参数
- 新增 `u_model_switch=13` 分支

### `reynold_pinn.py`
- Config: 新增 `u_model_switch=13`, `num_fourier_freq=4`, `embed_dim=64`
- `create_H_func`: 去掉了 step_type 开关, 固定使用 Hermite 三次过渡
- `create_pde_models`: 回归标准版, 去掉了 w_wedge/stop_gradient/point_weight
- `train_model`: 回归朴素版, 去掉 w_wedge 课程
- `main`: 简化调用链, `u_model_switch`/`num_freq`/`embed_dim` 从 Config 读取
- `layer_sizes`: `[2, 128, 128, 128, 128, 128, 2]` (6层)

---

## 使用

```bash
python reynold_pinn.py
```

默认使用新架构 (`u_model_switch=13`)。回退旧架构: 在 Config 中设 `u_model_switch=8`。

**注意**: 对比脚本 `compare_fem_pinn_final.py` 中硬编码了 `u_model_switch=8`。切换架构后需同步修改。
