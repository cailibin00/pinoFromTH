# 项目全景文档：基于PINN的螺旋槽推力轴承空化求解器

## 1. 项目概述

本项目使用 **物理信息神经网络（PINN, Physics-Informed Neural Network）** 求解带有 **空化效应** 的 **Reynolds方程**，应用于 **螺旋槽推力轴承** 的润滑分析。项目接收自他人，代码较为零散，但核心目标明确：用深度学习替代传统FEM（有限元法）来求解轴承润滑问题。

**核心技术栈**：Python + TensorFlow 2.x + 自研PINN框架TensorDiffEq

**研究问题**：螺旋槽推力轴承在高速旋转下的流体动压润滑与空化现象

---

## 2. 项目目录结构

```
pinoFromTH/
├── reynold_pinn.py                    ★ 主程序：PINN训练入口
├── compare_fem_pinn_final.py          ★ FEM vs PINN对比分析
├── compare_fem_pinn_iso_v16.py        另一版对比脚本（等温工况变体）
├── p_FBNS.txt                         FEM参考数据：压力场 (201×201网格)
├── g_FBNS.txt                         FEM参考数据：空化率场 (201×201网格)
│
├── tensordiffeq/                      ★ 自研PINN框架库
│   ├── __init__.py                    模块导出
│   ├── models.py                      ★ 核心模型：CollocationSolverND求解器类
│   ├── networks.py                    ★ 神经网络架构定义（多种PINN变体）
│   ├── fit.py                         训练循环（Adam + L-BFGS）
│   ├── boundaries.py                  边界条件定义（Dirichlet/Neumann/Periodic）
│   ├── domains.py                     计算域定义与配点生成
│   ├── utils.py                       工具函数（MSE、LHS采样等）
│   ├── PCGrad.py                      多任务梯度冲突解决
│   ├── sampling.py                    拉丁超立方采样
│   ├── optimizers.py                  优化器（L-BFGS）
│   ├── output.py                      训练输出与打印
│   └── helpers.py                     辅助函数
│
├── output/                            ★ 所有运行产物集中存放（git忽略）
│   ├── checkpoints/                   训练最佳权重（TensorFlow checkpoint）
│   │   └── epochs_best_model.*
│   ├── models/                        完整保存的 TensorFlow 模型
│   │   └── reynolds_pinn_N4900_iter80000/
│   │       ├── saved_model.pb
│   │       └── variables/
│   ├── figures/                       训练生成的可视化图片
│   │   └── reynolds_pinn_N4900_iter80000/
│   │       ├── *_pressure_contour.png
│   │       ├── *_cavitation_contour.png
│   │       ├── *_H_only.png
│   │       └── *_loss_log.png
│   └── comparison_results/            FEM vs PINN 对比分析输出
│       ├── fig1a~fig9_*.png          9组对比可视化
│       └── metrics.txt                误差指标汇总
│
└── doc/                               项目文档
    └── project_overview.md
```

---

## 3. 物理背景：Reynolds方程与空化

### 3.1 问题场景

推力轴承由两个平行圆盘组成，其中一个表面刻有**螺旋槽**（spiral groove）。当圆盘高速旋转时，流体被泵入槽内，形成动压润滑膜，从而承载轴向载荷。在高压高速工况下，润滑膜中会出现**空化（cavitation）**现象——局部压力降至蒸气压力以下，液体汽化。

### 3.2 几何参数

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 内径 | r_i | 47.0 mm | 轴承内边界 |
| 外径 | r_o | 52.0 mm | 轴承外边界 |
| 平衡膜厚 | h_i | 3.0 μm | 无槽区域的膜厚 |
| 周期数 | K | 6 | 360°分为6个周期 |
| 槽起始比 | R_d_1 | 1.043 × r_i | 槽起始径向位置 |
| 槽结束比 | R_d_2 | 2.212 × r_i | 槽终止径向位置 |
| 螺旋角 | α | 3.0° | 槽的螺旋方向 |
| 槽深比 | h_texture | 3.0 × h_i | 槽深度 |
| 槽宽比 | groove_ratio | 0.5 | 槽区与台区宽度比 |

### 3.3 工况参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 内径压力 p_i | 0.1 MPa | 内边界供油压力 |
| 外径压力 p_o | 1.5 × p_i | 外边界压力 |
| 动力粘度 η | 8.00×10⁻⁴ Pa·s | 润滑油粘度 |
| 转速 ω | 6000 rpm | 旋转速度 |

---

## 4. 核心数学模型详解

### 4.1 无量纲Reynolds方程

原始Reynolds方程经无量纲化后，在极坐标 (R, θ) 下表示为：

```
1/R · ∂/∂R [R·H³·∂P/∂R] + 1/R² · ∂/∂θ [H³·∂P/∂θ]
    - Λ·∂H/∂θ - Λ·∂(-γ·H)/∂θ = 0
```

**符号说明**：
- **P(R, θ)**：无量纲压力
- **γ(R, θ)**：空化率（cavitation fraction），γ=0表示全膜区，γ∈(0,1]表示空化区
- **H(R, θ)**：无量纲膜厚，由螺旋槽几何形状决定
- **Λ**：轴承数（bearing number），表征粘性力与惯性力之比：
  ```
  Λ = (6·η·ω·r_o²) / (h_i²·p_base)
  ```
- **R, θ**：无量纲径向和角度坐标

### 4.2 JFO空化模型（Jakobsson-Floberg-Olsson）

JFO模型是一种**质量守恒**的空化模型，是本文的核心物理约束：

- **全膜区（Full Film）**：P > 0, γ = 0
- **空化区（Cavitation）**：P = 0, γ > 0
- 两个区域通过**自由边界**（空化边界）分离

### 4.3 Fischer-Burmeister (FB) 互补条件

JFO模型的互补关系 `P ≥ 0, γ ≥ 0, P·γ = 0` 通过 **Fischer-Burmeister NCP函数** 光滑松弛：

```
P + γ - √(P² + γ²) = 0
```

该方程在P>0时自动迫使γ→0，在γ>0时自动迫使P→0，完美编码了互补条件。

### 4.4 螺旋槽膜厚函数 H(R, θ)

膜厚由螺旋槽几何定义，使用**sigmoid光滑过渡**避免不连续：

```
θ_sym(R) = ln(R/r_g) / tan(α) + θ_offset    ← 螺旋线方程

is_texture(R, θ) = σ((R-R_d_1)/ξ_R) · σ((R_d_2-R)/ξ_R) · Σ_periodic σ(...)

H(R, θ) = 1.0 × (1-is_texture) + (1 + h_texture/h_i) × is_texture
```

其中 σ 是sigmoid函数，ξ_R、ξ_θ是平滑参数。通过叠加多个周期偏移（0, ±2π/K, ±4π/K）实现周期性。

---

## 5. PINN神经网络架构

### 5.1 网络结构

使用 `u_model_switch=8`：`new_neural_period_polar_exactBC_two_output`

```
输入层:    [R, θ]  (2维)
    ↓
Coslayer_normalization  ★ 核心创新
    ├── 坐标归一化到 [-1, 1]
    └── 角度方向余弦傅里叶特征: cos(θ·K + φ)
    ↓
隐藏层:    [128, 128, 128, 128]
    每层使用U/V双分支 + 可学习混合权重:
    x_new = tanh(W_t·x ⊙ U + (1-W_t·x) ⊙ V)
    ↓
输出层:    [P_pred, γ_pred]  (2维)
    ↓
硬约束施加:
    P = g_func_P + σ_func_1 · P_pred
    γ = σ_func_2 · γ_pred
    ↓
最终约束:
    P = tanh(P)²         ← 非负约束
    γ = tanh(γ)²         ← 非负约束
```

### 5.2 关键设计点

1. **余弦傅里叶特征层（Coslayer_normalization）**
   - 将θ坐标通过 `cos(θ·π + φ)` 映射，天然满足**角度周期性**
   - 包含可训练参数：`kernel`, `phy`（相位）, `bias`

2. **硬约束边界条件（Hard BC Enforcement）**
   - 通过距离函数 `σ_func = (1-R²) / √((1-R)²+(1+R)²)` 在边界处自动归零
   - `g_func = atanh(线性插值(P_i, P_o))` 精确满足Dirichlet边界条件
   - 网络只需学习边值修正量，大幅加速收敛

3. **U/V双分支混合（Mixture of Branches）**
   - 每层维护两个独立分支U和V
   - 通过可学习的门控权重动态混合，增强表达能力

4. **输出约束**
   - 压力和空化率通过 `tanh²` 强制为非负值，物理合理性内建

---

## 6. 损失函数与训练策略

### 6.1 多任务损失函数

```
L_total = w₁·L_Reynolds + w₂·L_FB + w₃·L_BC_P + w₄·L_BC_γ
```

| 损失项 | 含义 | 公式 |
|--------|------|------|
| L_Reynolds | PDE残差（含τ稳定化） | MSE(f_model_FBNS, 0) |
| L_FB | Fischer-Burmeister互补条件 | MSE(P+γ-√(P²+γ²), 0) |
| L_BC_P | 压力Dirichlet边界条件 | MSE(P\|_boundary, P_target) |
| L_BC_γ | 空化率边界条件 | MSE(γ\|_boundary, 0) |

### 6.2 PDE残差中的τ稳定化项

在Reynolds方程残差中加入稳定化项，处理空化边界的数值不连续性：

```
f_p = PDE_standard + ∂²γ/∂θ² · τ · τ₂

其中:
    τ  = stop_gradient((|∂γ/∂θ| - ∂γ/∂θ) · ε)     ← 仅在空化增长率>0处激活
    τ₂ = stop_gradient((∂P/∂θ - |∂P/∂θ|) · ε)      ← 仅在压力下降处激活
    ε  = 0.1
```

该设计利用 `stop_gradient` 使稳定化项不影响梯度反传方向，只在物理上合理的区域（压力下降+空化增长）添加人工扩散。

### 6.3 PCGrad多任务梯度协调

直接对多任务损失加权求和会导致**梯度冲突**（不同任务的梯度方向相反）。PCGrad通过**梯度投影**解决这一问题：

```
对任意两个任务i, j的梯度g_i, g_j:
    if g_i · g_j < 0:  (梯度冲突)
        g_i = g_i - (g_i·g_j)/(g_j·g_j) · g_j   ← 将g_i投影到g_j的正交补
```

处理后各任务梯度不再相互对抗，训练更稳定。

### 6.4 自适应损失权重

使用**梯度幅值均衡法**自动调整各损失项的权重系数：

```
w_i_new = mean(|∇w·L_PDE|) / mean(|∇w·L_i|)
w_i = (1-α)·w_i_old + α·w_i_new     (指数移动平均, α=0.2)
```

### 6.5 RAD自适应配点细化

**RAD (Residual-based Adaptive Distribution)** 根据PDE残差大小自适应增加配点：

1. 在密集测试网格上计算PDE残差
2. 将残差的平方作为采样概率分布
3. 按概率权重采样更多配点，集中在残差大的区域
4. 每轮训练后执行RAD，多轮迭代（NL_train=4轮）

### 6.6 训练流水线

```
阶段1: LR = [1e-3 → 1e-4 → 1e-5], NL_train×RAD 细化
阶段2: LR = [1e-4 → 1e-4 → 1e-5], NL_train×RAD 细化
阶段3: LR = [1e-5 → 1e-4 → 1e-5], NL_train×RAD 细化
阶段4: LR = [1e-5 → 1e-5 → 1e-6], NL_train×RAD 细化
────────────────────────────────────────────
总计: 4阶段 × NL_train轮 × N_train步 = 4×4×5000 = 80000次迭代
```

每5000步为1轮，每轮结束后执行RAD细化配点分布。

---

## 7. FEM对比验证

`compare_fem_pinn_final.py` 将PINN预测结果与传统有限元法（FEM）的参考解进行系统对比，输出到 `output/comparison_results/`：

### 7.1 FEM参考数据
- **格式**：`p_FBNS.txt` 和 `g_FBNS.txt`（位于项目根目录），每行 `[R, θ, value]`
- **分辨率**：201×201 = 40,401个网格点
- **来源**：FBNS (Fischer-Burmeister NCP Solver) 有限元求解器
- **PINN权重**：从 `output/checkpoints/epochs_best_model` 加载

### 7.2 误差指标（训练结果）

| 指标 | 数值 | 说明 |
|------|------|------|
| P相对L2误差 | 1.39×10⁻² | 压力全局相对误差~1.4% |
| P相对L∞误差 | 7.22×10⁻² | 压力最大相对误差~7.2% |
| G相对L2误差 | 1.18×10⁻¹ | 空化率全局相对误差~11.8% |
| 互补条件违反量 | 3.20×10⁻³ | P·γ最大值（应→0） |
| 空化IoU | 0.271 | 空化区域交并比 |
| 空化Dice | 0.426 | 空化区域Dice系数 |

### 7.3 关键发现
- 压力场精度较高（~1.4% L2误差）
- 空化率预测精度略低（~11.8%），空化边界吻合度有限（IoU≈0.27）
- JFO互补条件被基本满足（P·γ≈3.2×10⁻³）

---

## 8. 框架设计模式

TensorDiffEq采用**面向对象PINN框架**设计：

```
DomainND          ← 计算域定义（范围+网格密度+配点生成）
    ↓
dirichletBC       ← 边界条件（自动生成边界配点坐标）
    ↓
CollocationSolverND  ← 核心求解器
    ├── compile()    绑定网络/域/边界/损失函数
    ├── fit()        执行训练循环
    ├── RAD_FB()     残差自适应配点细化
    └── predict()    推理预测
```

### 网络变体设计（u_model_switch）

该框架预留了多种PINN网络变体（switch 1~12），针对不同物理约束场景：
- **Switch 1**：标准MLP（无约束）
- **Switch 4**：单输出+硬约束Dirichlet BC+非负输出
- **Switch 6**：双输出+硬约束Dirichlet BC
- **Switch 8**（本项目使用）：极坐标+周期性+硬约束BC+双输出
- **Switch 11~12**：带纹理函数的双输出变体

---

## 9. 代码入口与运行

所有运行产物统一输出到 `output/` 目录（已加入 `.gitignore`，不纳入版本控制）。

### 训练模型
```bash
python reynold_pinn.py
```
输出：
- 训练好的模型：`output/models/reynolds_pinn_N4900_iter80000/`
- 最佳权重：`output/checkpoints/epochs_best_model.*`
- 可视化图片：`output/figures/reynolds_pinn_N4900_iter80000/` 下的压力轮廓图、空化轮廓图、膜厚图、损失曲线

### 对比分析
```bash
python compare_fem_pinn_final.py
```
输出（均在 `output/comparison_results/` 下）：
- `fig1a~fig9_*.png`：9组对比可视化
- `metrics.txt`：数值误差指标

---

## 10. 技术亮点总结

1. **JFO空化模型的PINN实现**：将互补条件编码为FB函数损失 + PDE残差中嵌入稳定化项
2. **硬约束边界条件**：通过距离函数和atanh变换在NN结构中精确嵌入物理边界
3. **周期性自动满足**：Coslayer_normalization通过傅里叶特征天然编码角度周期性
4. **PCGrad梯度协调**：解决多任务学习中PDE残差与互补条件的梯度冲突
5. **自适应配点策略**：RAD在高误差区域自动增加采样密度
6. **完整的FEM对比验证**：9组可视化 + 多维度误差指标

---

> **文档版本**: v1.1
> **编写日期**: 2026-06-04
> **修订记录**:
> - v1.1: 重构输出目录为 `output/{checkpoints,models,figures,comparison_results}`，同步更新代码与文档
> - v1.0: 初始版本
> **原始作者**: [叶萌]（代码注释中标注）
