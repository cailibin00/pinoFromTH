# 模型架构改进日志

> **日期**: 2026-06-05
> **基于**: 原始代码 v1.1 (叶萌)
> **改进依据**: 对原模型四个核心瓶颈的针对性修改

---

## 改进总览

| 问题 | 根因 | 修改 |
|------|------|------|
| 边界处理 | `g_func` 只有3自由度(atanh-线性+Hermite) | `bc_switch=1`: MLP_g(R) 可学习基线 + plateau_σ(R) |
| 膜厚平滑 | sigmoid 过渡区模糊, 且 ∂H/∂θ 梯度劫持全局更新 | ABC组合控梯策略 |
| 输入编码 | R 只有线性投影, θ↔R 加法耦合 | R-MLP + θ-MLP 独立 Fourier 编码, concat |
| 测评稳定 | 近零分母导致相对误差爆炸 | (见后续单独改进) |

### 开关总表

| 配置项 | 值 | 含义 |
|--------|----|------|
| `u_model_switch` | `13` | 新 Fourier 解耦架构 |
| `bc_switch` | `1` | 改进硬 BC (MLP_g + plateau σ) |
| `bc_switch` | `2` | 纯软约束, 放弃硬 BC |
| `h_step_type` | `'hermite'` | C¹ 分段三次膜厚过渡 (推荐) |
| `h_step_type` | `'relu'` | C⁰ 锐利过渡 |
| `h_step_type` | `'sigmoid'` | C∞ 平滑 (旧行为) |

---

## 1. 输入编码：Fourier 解耦架构

### 问题

旧架构 (`u_model_switch=8`) 的 Coslayer_normalization:
- R 坐标只做 `R * kernel_1[0]` —— **单一线性投影**, 表达能力极弱
- θ 坐标做 `cos(θ·K + φ)` —— 周期性编码良好
- R 和 θ 通过**加法融合**: `R*kernel_1 + cos(θ)*kernel_2 + bias`
  — 破坏了信息解耦, 无法选择性忽略某个坐标

### 修改

新增 `new_neural_fourier_decoupled()` ([networks.py:806](tensordiffeq/networks.py#L806)):

```
输入 [R, θ]
    │
    ├── R → normalize to [-1,1]
    │       → Fourier [sin(2ⁱπR), cos(2ⁱπR)] for i=0..L-1  (NeRF-style)
    │       → MLP_R [2L → 32 → embed_dim]  → R_embed
    │
    └── θ → normalize to [-1,1]
            → Fourier [sin(2ⁱπθ), cos(2ⁱπθ)] for i=0..L-1
            → MLP_θ [2L → 32 → embed_dim]  → θ_embed

    Concat(R_embed, θ_embed)  →  [2×embed_dim]

    Main Network (U/V branching):
        x_U, x_V = Dense(width)(concat)
        for each layer:
            gate = tanh(Dense(x))
            x = gate * x_U + (1-gate) * x_V
```

**新增 Config 参数**:
- `num_fourier_freq = 4` (L=4, 每坐标 8 个 Fourier 特征)
- `embed_dim = 64` (R/θ 独立编码到 64 维, concat 后 128 维进入主干)

### 效果

- R 方向获得高频表达能力 (Fourier 频率最高 2³π = 8π)
- θ 方向保留周期性 (cos feature 天然周期)
- concat 保持信息解耦, 网络可选择性关注某一坐标
- 主干网络宽度保持 128 (与旧模型一致)

---

## 2. 边界条件：MLP_g(R) + plateau σ(R)

### 问题

旧硬约束:
```
P_final = tanh²( atanh(线性插值(√P_i, √P_o, R))     ← g_func_1: 1自由度(直线)
                + x₁(R+1)((R-1)/(-2))² + x₂(R-1)((R+1)/2)² ← g_func_2: 2自由度(Hermite)
                + σ(R)·NN_output )                              ← 边界处σ→0,NN被清零
```
- `g_func_1 + g_func_2` 总共只有 **3 个自由度**, 且被限制在 atanh-√ 空间
- σ(R) 用指数型 `1-e^{p(-1-R)}`, 在边界附近衰减极快 → **"边界粘滞区"**
- 真实解在 atanh-√ 空间不可能是直线+单峰 → 边界附近 g_func 错了, NN 也改不了

### 修改: bc_switch=1 (改进硬 BC)

**g_mlp(R)** ([networks.py:892-898](tensordiffeq/networks.py#L892-L898)):

```
g_mlp:  R_norm → Dense(8, tanh) → Dense(8, tanh) → Dense(1)  →  g_func
参数数: (1×8+8)+(8×8+8)+(8×1+1) = 97 个可学习参数
```

- 放弃 atanh-√ 直线先验, 让 MLP 从数据中学习 R→P_baseline 映射
- 与主干网络共享反向传播, 不需单独训练

**plateau σ(R)** ([networks.py:900-905](tensordiffeq/networks.py#L900-L905)):

```
transition = 0.03  (归一到[-1,1]后, 占域宽 3%)

t_left  = clip((R_norm+1)/transition,  0, 1)
t_right = clip((1-R_norm)/transition,  0, 1)

σ(R) = (3·t_left² - 2·t_left³) × (3·t_right² - 2·t_right³)
       ───── Hermite cubic ────     ───── Hermite cubic ────
```

σ(R) 的形状:
```
 1.0 ───────────────────────────┐         ┌─── 1.0
                                 │         │
                                /           \
                               /             \
 0.0 ─────────────────────────╱               ╲─────── 0.0
     R=-1           R=-1+0.03              R=1-0.03    R=1
```

- 中央 94% 区域 σ≡1 → NN 有充分自由修正 g_mlp
- 只在距边界 3% 内衰减到 0 → 硬满足边界值
- Hermite cubic = C¹ 光滑 → 导数连续, auto-diff 不出 NaN

### 修改: bc_switch=2 (纯软约束)

```
P = tanh(NN_P_output)²
γ = tanh(NN_γ_output)²
```

- 完全抛弃硬约束范式
- 边界值由 BC 损失项 + PCGrad 自适应权重 + 高初始化权重 保证
- 适用场景: 当域边界不规则或边界条件复杂时

---

## 3. 膜厚梯度：ABC 组合控梯策略

### 问题

PDE 残差中 `∂H/∂θ` 以裸量出现 (前乘 Λ≈50):

```
part_3_1 = -Λ·∂H/∂θ          ← 在槽边界处 ≈ 1800
part_1,2 = Poiseuille 扩散项  ← ≈ 1~10

→ 全局梯度由楔形项绝对主导 (差3个数量级)
→ 优化器不计代价地"抹平" H 的梯度 → 忽略空化/扩散
```

### 修改

三招组合 ([reynold_pinn.py:175-239](reynold_pinn.py#L175-L239)):

**A. `stop_gradient(H)` 截断楔形梯度** (L200-201):
```python
H_for_wedge = tf.stop_gradient(H)  # H数值正确, 但梯度不反传
part_3_1 = -Λ · ∂(H_for_wedge)/∂θ  # 对网络的 ∂L/∂w 中此项=0
```
- 楔形项仍然驱动 PDE 残差 (物理正确)
- 但不再劫持网络权重的更新方向

**B. `w_wedge` 课程权重** (L189-190, 310-320):
```python
w_wedge:  阶段1=0.01 → 阶段2=0.34 → 阶段3=0.67 → 阶段4=1.0
part_3_1 *= w_wedge
```
- 前20000步: 几乎是纯 Poiseuille 扩散 (学光滑压力场)
- 20000-40000步: 逐步引入楔形效应
- 60000-80000步: 完整 PDE

**C. 逐点权重压制槽边界** (L222-231):
```python
grad_H_theta = ∂H/∂θ                         # 原始 H 的梯度
mean_grad = mean(|∂H/∂θ|)                     # ∼整个域的均值
point_weight = 1 / (1 + |∂H/∂θ| / mean_grad) # 槽边界≈0.001, 其他≈1
f_p = point_weight * f_p_raw
```
- 槽边界点 (|∂H/∂θ| ≫ mean) →权重→0, loss 贡献被压制
- 平滑区域点 →权重≈1, loss 贡献不受影响
- 使用 `stop_gradient(weight)` 避免引入二阶梯度

---

## 修改的文件

### `tensordiffeq/networks.py`

- **新增**: `new_neural_fourier_decoupled()` (L806-930) — Fourier 解耦架构
  - 输入: (N,2), 输出: [P(N,1), γ(N,1)]
  - 包含独立 R-MLP + θ-MLP + 主干 U/V 网络
  - `bc_switch=1`: MLP_g(R) + plateau σ(R) + tanh²
  - `bc_switch=2`: 纯 tanh² (无硬约束)

### `tensordiffeq/models.py`

- **修改**: `compile()` 签名新增 `bc_switch`, `num_freq`, `embed_dim` (L24)
- **新增**: `u_model_switch=13` 分支 (L103-114)
  - 存储 `self.bc_switch`, `self.num_freq`, `self.embed_dim`
  - 调用 `new_neural_fourier_decoupled()`

### `reynold_pinn.py`

- **Config 类**: 新增 10 个参数 (L67-86)
  - `u_model_switch=13`, `bc_switch=1`
  - `num_fourier_freq=4`, `embed_dim=64`
  - `h_step_type='hermite'`
  - `w_wedge_init/final`, `use_stop_gradient_H`, `use_point_weight`

- **`create_H_func()`** (L138-199): 新增三种过渡类型
  - `_step_fn(x)`: 根据 `h_step_type` 选择 hermite/relu/sigmoid
  - Hermite: `clip(x,0,1)` → `3t²-2t³`
  - RELU: `clip(x,0,1)` (C⁰)

- **`create_pde_models()`** (L201-248): 返回 3 元组
  - 新增 `w_wedge_var` (trainable=False)
  - 方法A: `stop_gradient(H)` 用于楔形项
  - 方法B: `w_wedge_var` 缩放楔形项
  - 方法C: `point_weight` 逐点降权
  - 返回 `set_w_wedge(new_val)` 调度函数

- **`train_model()`** (L307-350): 新增 w_wedge 课程
  - 4 阶段线性插值: `[0.01, 0.34, 0.67, 1.0]`
  - 每阶段开始时打印 `w_wedge` 值

- **`main()`** (L497-563):
  - 使用 `cfg.u_model_switch` (不再硬编码 8)
  - 传递 `bc_switch`, `num_freq`, `embed_dim` 到 `compile()`
  - 挂载 `model.set_w_wedge` 供 train_model 使用

### `doc/update.md` (新建)

- 本文件

---

## 运行方式

```bash
# 使用新架构 (默认 bc_switch=1, h_step_type='hermite')
python reynold_pinn.py

# 若要回退旧行为, 修改 Config 中的:
#   u_model_switch = 8     (旧极坐标硬BC)
#   h_step_type = 'sigmoid' (旧膜厚)
```

**注意**: 对比脚本 `compare_fem_pinn_final.py` 中硬编码了 `u_model_switch=8`。使用新模型训练后, 需将其中 `u_model_switch=8` 改为 `13`, 并添加相应参数, 否则加载权重会因架构不匹配而失败。

---

## 向后兼容

- 所有原 switch (1-12) 和原有函数**不受影响**
- `create_H_func` 的 `cfg` 参数可传入旧 Config (无 `h_step_type` 属性时会 fallback 到 `'sigmoid'`)
- `create_pde_models` 的 `cfg=None` 默认关闭 ABC 策略 (旧行为)
- `compile()` 的新参数均有默认值, 旧代码不传也能运行
