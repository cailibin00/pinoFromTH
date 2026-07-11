# 当前训练策略文档

> 最后更新：2026-07-11（移除 PCGrad + 自适应权重，引入 CosineAnnealingLR）

---

## 目录

1. [概览：训练流水线](#1-概览训练流水线)
2. [单步训练 (train_step)](#2-单步训练-train_step)
3. [损失函数构成](#3-损失函数构成)
4. [优化器：Adam](#4-优化器adam)
5. [学习率调度：CosineAnnealingLR](#5-学习率调度cosineannealinglr)
6. [四阶段训练](#6-四阶段训练)
7. [RAD：残差驱动的自适应配点细化](#7-rad残差驱动的自适应配点细化)
8. [L-BFGS 二阶段精调（当前关闭）](#8-l-bfgs-二阶段精调当前关闭)
9. [最佳模型保存策略](#9-最佳模型保存策略)
10. [完整超参数速查表](#10-完整超参数速查表)
11. [与旧版本 (TF port) 的差异](#11-与旧版本-tf-port-的差异)

---

## 1. 概览：训练流水线

```
main()
  │
  ├─ 1. Config 初始化
  ├─ 2. 物理参数计算 (Λ=604, P_i, P_o, ...)
  ├─ 3. 膜厚函数 H(R,θ) 创建
  ├─ 4. PDE 残差函数 f_model_FBNS + f_model_FB 创建
  ├─ 5. 配点生成 (LHS + 结构化网格 + 沟槽边界点) → ~8000 点
  ├─ 6. 模型创建 + compile()
  │
  └─ train_model()
       │
       └─ for stage in [1, 2, 3, 4]:           ← 4 个阶段
            │
            └─ for round in [1, 2, 3, 4]:       ← 每阶段 4 轮
                 │
                 ├─ CosineAnnealingLR 初始化     ← lr: init_lr → 1e-6
                 ├─ fit(tf_iter=5000)            ← Adam × 5000 步
                 │    └─ 每步: train_step()
                 │         ├─ update_loss_seperate()  → [L_Reynolds, L_BC, L_FB]
                 │         ├─ loss_total = sum(losses)
                 │         ├─ loss_total.backward()
                 │         └─ optimizer.step()
                 │              └─ scheduler.step()   ← 余弦衰减
                 │
                 └─ RAD_FB()                    ← 残差驱动配点细化
```

**一条完整的训练 run 的算量**：

```
总 Adam 迭代 = 4 stages × 4 rounds × 5,000 steps = 80,000 步
```

当前 L-BFGS 关闭 (`newton_iter=0`)，不参与训练。

---

## 2. 单步训练 (train_step)

代码位置：[`torch_pinn/models.py:324-335`](../torch_pinn/models.py#L324-L335)

```python
def train_step(self):
    self.u_model.train()
    self.tf_optimizer.zero_grad()

    loss_all = self.update_loss_seperate()   # 计算 3 个损失
    loss_total = sum(loss_all)               # 直接求和

    loss_total.backward()                    # 标准反向传播
    self.tf_optimizer.step()                 # Adam 更新

    return loss_total.detach(), [l.detach() for l in loss_all]
```

### 关键设计决策

| 特性 | 当前状态 | 说明 |
|------|----------|------|
| **梯度冲突消解** | ❌ 无 PCGrad | 旧版 PCGrad 已移除 |
| **自适应权重** | ❌ 无 | 旧版 ComputeSum_weight 已移除 |
| **损失组合方式** | `sum(losses)` | 简单的无加权直接求和 |
| **所有配点全量使用** | ✅ full-batch | `batch_size=None`，一次前传在全部 ~8000 点上计算 |

### train_step 的完整计算图

```
train_step()
  │
  ├─ update_loss_seperate()
  │    │
  │    ├─ _get_batch_X_f()                          ← 取全量配点 [N_f, 2]
  │    ├─ R, theta = X_f[:,0], X_f[:,1]             ← 需要梯度 (autograd)
  │    │
  │    ├─ [1] L_Reynolds
  │    │    └─ f_model_FBNS(u_model, R, theta)       ← 雷诺方程残差 f_p
  │    │         ├─ p, gamma = u_model(R,theta)       ← NN 前传
  │    │         ├─ p_R, p_theta = autograd(p)        ← 一阶导数
  │    │         ├─ part_1 = ∂(R·H³·p_R)/∂R / R      ← 二阶导数
  │    │         ├─ part_2 = ∂(H³·p_θ)/∂θ / R²        ← 二阶导数
  │    │         ├─ part_3_1 = -Λ·∂H/∂θ               ← 解析计算
  │    │         ├─ part_3_2 = Λ·∂(γ·H)/∂θ            ← 一阶导数
  │    │         ├─ stab = ∂²γ/∂θ²·τ·τ₂               ← 稳定项
  │    │         └─ loss = MSE(f_p, 0)                ← 残差 → 损失
  │    │
  │    ├─ [2] L_BC
  │    │    └─ update_loss_bcs()
  │    │         └─ mean(MSE(p_pred_at_boundary, BC_val))
  │    │
  │    └─ [3] L_FB
  │         └─ update_loss_JFO_term_interact()
  │              └─ MSE(p+γ-√(p²+γ²), 0)
  │
  ├─ loss_total = L_Reynolds + L_BC + L_FB           ← 直接求和
  │
  └─ loss_total.backward() → optimizer.step()
```

### 梯度流的量级估算

```
L_Reynolds ≈ 10⁵ ~ 10⁶   →  |∂L/∂θ| ≈ 10³ ~ 10⁵   (二阶 autograd 放大)
L_BC       ≈ 10⁻⁴ ~ 10⁻³  →  |∂L/∂θ| ≈ 10⁻³ ~ 10⁻¹
L_FB       ≈ 10⁻³         →  |∂L/∂θ| ≈ 10⁻² ~ 10⁰
```

**L_Reynolds 的梯度完全主导了参数更新**。这在实际中可能利大于弊：雷诺方程是核心物理约束，需要最强优化信号。但代价是 JFO 互补条件和 BC 的信号被淹没。

---

## 3. 损失函数构成

代码位置：[`torch_pinn/models.py:282-313`](../torch_pinn/models.py#L282-L313)

### 3.1 完整损失列表

| # | 名称 | 代码位置 | 公式 | 典型值 |
|---|------|----------|------|--------|
| 1 | **L_Reynolds** | `update_loss_res()` → `f_model_FBNS` | `MSE(f_p, 0)` | 10⁵ ~ 10⁶ |
| 2 | **L_BC** | `update_loss_bcs()` | `mean(MSE(p_in, P_i) + MSE(p_out, P_o))` | 10⁻⁴ ~ 10⁻³ |
| 3 | **L_FB** | `update_loss_JFO_term_interact()` | `MSE(p+γ−√(p²+γ²), 0)` | ~10⁻³ |

### 3.2 损失计算流程

```
update_loss_seperate()
  │
  ├─ for f_model in self.f_model_list:         ← 当前只有一个: f_model_FBNS
  │    └─ update_loss_res(f_model, R, theta)   → [L_Reynolds]
  │
  ├─ update_loss_bcs()                         → [L_BC]
  │    └─ 遍历 self.bcs (内径 + 外径 Dirichlet)
  │         └─ MSE(u_model(bc_input)[0], bc_val)
  │         └─ 返回 [mean(所有BC损失)]
  │
  └─ if two_output:                            ← ✅ True
       └─ update_loss_JFO_term_interact()      → L_FB (标量，append 到列表)
            └─ 在全量配点上计算 FB 残差
```

**`L_BC` 的特殊性**：边界条件通过构造精确满足（`sigma_func` + `g_func`），所以 `L_BC` 天然很小。详见 [`loss.md`](loss.md) 附录。

### 3.3 与 BC 硬执行的配合

网络输出构造（[networks.py:416-422](../torch_pinn/networks.py#L416-L422)）：

```
P = tanh(g_func_hermite + sigma_distance × NN_output)²
γ = tanh(sigma_distance × NN_output)²
```

- 在边界处 `sigma=0`，`g_func=BC值` → **P 精确等于 BC 值**
- 这种硬执行使得 `L_BC` 只是对 BC 精度的一个弱验证，不是主动约束

---

## 4. 优化器：Adam

代码位置：[`torch_pinn/models.py:157-161`](../torch_pinn/models.py#L157-L161)

```python
self.tf_optimizer = torch.optim.Adam(
    self.u_model.parameters(),
    lr=0.001,              # 初始学习率（随后被 scheduler 覆盖）
    betas=(0.99, 0.999)
)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `lr` | 动态（见调度器） | compile 时设 0.001，随后被覆盖 |
| `β₁` | 0.99 | 一阶动量系数 |
| `β₂` | 0.999 | 二阶动量系数 |
| `ε` | 默认 1e-8 | |
| `weight_decay` | 0 | 无 L2 正则化 |

**选择 (0.99, 0.999) 而非默认 (0.9, 0.999) 的原因**：更高的 β₁ 意味着更平滑的梯度估计，适合损失曲面上存在高频振荡的 PINN 训练。

---

## 5. 学习率调度：CosineAnnealingLR

代码位置：[`reynold_pinn_torch.py:312-322`](../reynold_pinn_torch.py#L312-L322)

### 5.1 公式

```
lr(t) = eta_min + 0.5 × (init_lr − eta_min) × (1 + cos(π × t / T_max))
```

其中 t ∈ [0, T_max] 是当前迭代步数，T_max = 5000。

### 5.2 余弦曲线示意

```
lr
│
│  init_lr ─┐
│            ╲
│             ╲
│              ╲____
│                   ╲___  cosine decay
│                        ╲___
│                             ╲____
│                                   ╲____
│  eta_min ──────────────────────────────
│
└──────────────────────────────────────→ 迭代步数
0                                  T_max=5000
```

### 5.3 每阶段初始学习率

| 阶段 | init_lr | eta_min | T_max | 衰减幅度 |
|------|---------|---------|-------|---------|
| Stage 1 | **1×10⁻³** | 1×10⁻⁶ | 5000 | 1000× |
| Stage 2 | **1×10⁻⁴** | 1×10⁻⁶ | 5000 | 100× |
| Stage 3 | **1×10⁻⁵** | 1×10⁻⁶ | 5000 | 10× |
| Stage 4 | **1×10⁻⁵** | 1×10⁻⁶ | 5000 | 10× |

### 5.4 为什么选择余弦退火？

| 对比维度 | 旧版：分段常数 LR | 新版：余弦退火 |
|----------|-------------------|---------------|
| 衰减方式 | 阶梯式突变 (20k→40k) | 光滑连续衰减 |
| 训练稳定性 | 阶梯处可能有 loss 跳变 | 平滑过渡 |
| 逃离局部最优 | 下降后固定不动 | 持续下降 + 每轮重置 |
| 可复现性 | 依赖 epoch 计数 | 纯 cos 函数，确定性 |

### 5.5 调度器何时被 step

在 `fit()` 的每个 Adam epoch 之后：

```python
for epoch in range(tf_iter):
    loss_value, loss_all = obj.train_step()   # optimizer.step() 在内
    if scheduler is not None:
        scheduler.step()                      # ← lr 衰减一步
```

---

## 6. 四阶段训练

代码位置：[`reynold_pinn_torch.py:298-340`](../reynold_pinn_torch.py#L298-L340)

### 6.1 阶段设计理念

```
Stage 1: lr 1e-3→1e-6  │  "探索"  大 LR 快速接近解
Stage 2: lr 1e-4→1e-6  │  "定位"  中等 LR 定位空化边界
Stage 3: lr 1e-5→1e-6  │  "精调"  小 LR 细化压力场
Stage 4: lr 1e-5→1e-6  │  "收敛"  最小 LR 最终收敛
```

每个阶段的 LR 起点比前一阶段低 10×（Stage 3→4 除外），形成**递进缩小**的搜索范围。

### 6.2 每阶段内部：4 轮 RAD

```
for round in [1, 2, 3, 4]:
    ┌─────────────────────────────────────┐
    │ Adam × 5000 步 (CosineAnnealingLR)  │  ← 模型优化
    └─────────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────────┐
    │ RAD_FB 配点细化                      │  ← 数据优化
    │  └─ 在当前最优模型上评估残差          │
    │  └─ 残差大 → 高采样概率               │
    │  └─ 新点加入训练集                    │
    └─────────────────────────────────────┘
```

**RAD 在每轮 Adam 训练之后执行**，这意味着模型在先一轮优化中已经学到了当前配点分布下的局部最优，然后 RAD 在"模型不懂的地方"加入新点，迫使下一轮学习更难的区域。

### 6.3 配点增长

```
初始配点: N_f_true ≈ 8259 (4900 LHS + 沟槽边界点)

每轮 RAD 增加:
  f_model_FBNS:  0.03 × N_f_true ≈ 248 点
  f_model_FB:    0.01 × N_f_true ≈ 83 点
  ─────────────────────────────────────
  每轮增加: ~331 点
  每阶段增加: 4 × 331 ≈ 1324 点
  总计增加: 4 × 1324 ≈ 5296 点

最终配点数: 8259 + 5296 ≈ 13,555 点
```

---

## 7. RAD：残差驱动的自适应配点细化

代码位置：[`torch_pinn/models.py:377-438`](../torch_pinn/models.py#L377-L438)

### 7.1 算法步骤

```
RAD_FB(f_model_list, N_raw, num_add_points_test, num_add_points, k, c)

Step 1: 保留原始 N_raw 个配点（丢弃之前 RAD 添加的点）
Step 2: 随机生成 num_add_points_test 个候选配点 (~10× N_raw)
Step 3: 对每个 f_model:
    a. 在当前最优模型上计算 PDE 残差 f²
    b. 计算采样概率: p(x) = f(x)^(2k) / mean(f^(2k)) + c
       (当前设置 k=1, c=1e-16)
    c. 按概率 p(x) 抽样 num_add_points[i] 个新点
    d. 将新点加入 self.domain.X_f
```

### 7.2 概率公式

```
err_eq(x) = f(x)² / (mean(f²) + ε) + c

采样概率 P(x) = err_eq(x) / Σ err_eq
```

- **k=1**：使用残差的平方（f²）作为权重。残差大的点权重高
- **c=1e-16**：微小常数确保无残差的区域也有极小概率被采样（保持均匀探索）
- 除以 `mean(f²)`：自归一化，防止残差整体增大时概率分布退化

### 7.3 RAD 的本质

RAD 是一种**重要性采样**策略：

> "给模型考它不会的题" —— PDE 残差大的区域自动获得更多训练样本

这类似于：
- 有限元中的 **h-自适应网格细化**（误差估计 → 局部加密网格）
- 强化学习中的 **优先经验回放**（TD 误差大 → 高采样概率）

---

## 8. L-BFGS 二阶段精调（当前关闭）

代码位置：[`torch_pinn/fit.py:76-123`](../torch_pinn/fit.py#L76-L123)

```python
# 当前配置（不执行）
if newton_iter > 0:   # newton_iter = 0 → 跳过
    ...
```

### 8.1 L-BFGS 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 0.8 | 线搜索的初始步长 |
| `history_size` | 50 | 存储最近 50 步的曲率信息 |
| `tolerance_change` | 1e-12 | 参数变化 < 1e-12 时停止 |
| `line_search_fn` | `strong_wolfe` | 强 Wolfe 条件确保充分下降 |

### 8.2 Adam → L-BFGS 的逻辑

```
Adam:  一阶方法，带有动量，适合非凸优化的初期探索
        ↓ 80,000 步后
L-BFGS: 准牛顿法，使用曲率信息，适合接近凸区域的精调
        可以在 50~200 步内将损失再降 1~2 个数量级
```

**为什么当前关闭？** 因为当前模型（残差连接 + 深输出头）参数量大（368k），L-BFGS 的 Hessian 近似开销与参数量成正比，可能导致显存不足或收敛慢。Adam 80,000 步后可以先评估结果，再决定是否开启 L-BFGS。

---

## 9. 最佳模型保存策略

代码位置：[`torch_pinn/fit.py:67-71`](../torch_pinn/fit.py#L67-L71)

```python
# 每 100 个 epoch
if epoch % 100 == 0:
    if loss_value.item() < obj.loss_value_min:
        obj.loss_value_min = loss_value.item()
        obj.save_weights(obj.best_weights_path)
```

- **检查频率**：每 100 步
- **保存条件**：总损失 `loss_total` 创历史新低
- **保存内容**：模型权重 + 优化器状态 + 损失历史
- **保存路径**：`output_torch1/checkpoints/epochs_best_model.pt`

---

## 10. 完整超参数速查表

### 10.1 训练超参数

| 参数 | 值 | 代码位置 |
|------|-----|----------|
| 总阶段数 | 4 | `train_model()` |
| 每阶段轮数 (NL_train) | 4 | `Config.NL_train` |
| 每轮 Adam 步数 (N_train) | 5000 | `Config.N_train` |
| **总 Adam 步数** | **80,000** | 4×4×5000 |
| 初始配点数 (N_f) | 4900 | `Config.N_f` |
| 实际初始配点 | ~8259 | 含沟槽边界点 |
| 每轮 RAD 新增 (f_model) | 3% | `ratio_RAD_list[0]` |
| 每轮 RAD 新增 (FB) | 1% | `ratio_RAD_list[1]` |
| RAD 候选点倍数 | 10× | `num_add_points_test` |
| RAD 权重幂次 (k) | 1 | `RAD_FB(k=1)` |
| RAD 均匀偏移 (c) | 1e-16 | `RAD_FB(c=1e-16)` |
| L-BFGS 迭代 | 0（关闭） | 调用时 `newton_iter=0` |

### 10.2 优化器超参数

| 参数 | 值 |
|------|-----|
| Optimizer | Adam |
| β = (β₁, β₂) | (0.99, 0.999) |
| LR Scheduler | CosineAnnealingLR |
| T_max | 5000 (per round) |
| eta_min | 1×10⁻⁶ |
| Stage 1 init_lr | 1×10⁻³ |
| Stage 2 init_lr | 1×10⁻⁴ |
| Stage 3 init_lr | 1×10⁻⁵ |
| Stage 4 init_lr | 1×10⁻⁵ |

### 10.3 日志 & 保存频率

| 事件 | 频率 | 内容 |
|------|------|------|
| 终端打印 | 每 500 步 | L_Reynolds, L_FB, lr |
| 进度条更新 | 每 10 步 | 当前总损失 |
| 最佳模型保存 | 每 100 步（如果最优） | model + optimizer state |
| 最终模型保存 | 训练结束 | 完整 config + weights |
| 损失 JSON 导出 | 训练结束 | loss/epoch history |

---

## 11. 与旧版本 (TF port) 的差异

| 组件 | 旧版 (TF port) | 当前版本 | 影响 |
|------|---------------|----------|------|
| **梯度计算** | PCGrad（投影冲突梯度） | 简单 `sum.backward()` | L_Reynolds 梯度主导 |
| **损失权重** | ComputeSum_weight（自适应 EMA） | 无加权（等权求和） | 无自动量级平衡 |
| **MTL 加权** | log-variance weighting（可选） | ❌ 移除 | — |
| **LR 调度** | 分段常数（20k/40k 阶梯） | CosineAnnealingLR（光滑） | 更稳定衰减 |
| **BC 损失** | 可选开关 (Boundary_true) | 始终包含 | BC 总是参与训练 |
| **自适应权重更新** | 每步 EMA 更新 | ❌ 移除 | — |

### 为什么做这些简化？

1. **PCGrad 的计算开销**：每步需要 3 次独立的 `autograd.grad()` + flatten + 投影，在 368k 参数上代价高昂
2. **自适应权重的脆弱性**：梯度比值 EMA 需要仔细调 α，对初始化和噪声敏感
3. **余弦退火的优越性**：相比分段常数，余弦衰减在 PINN 训练中通常给出更稳定的收敛曲线
4. **简化调试**：移除复杂机制后，训练行为更容易理解和诊断

### 如果损失不收敛，可以逐步加回的机制（由简到繁）

```
当前: sum(losses) + CosineAnnealingLR
  │
  ├─ 第一步: 加入手动固定权重  w₁·L_Reynolds + w₂·L_BC + w₃·L_FB
  │           (手动调整 wᵢ 来平衡量级)
  │
  ├─ 第二步: 加入 PCGrad（梯度冲突消解）
  │
  └─ 第三步: 加入自适应权重（自动调整 wᵢ）
```

---

*文档生成日期：2026-07-11*
