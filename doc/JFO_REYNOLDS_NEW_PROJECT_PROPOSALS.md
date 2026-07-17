# JFO 空化 Reynolds 问题：失败根因与两套从零 PINN 方案

> 文档状态：新项目架构决策稿，不是对旧模型的增量修补。
>
> 范围：只审计 `0504` TensorFlow 版本、`output_resF_silu_small`、`output_tanh_resF` 和相关 Git 历史；不考虑 torch、raw、PCGrad、PIKAN。
>
> 关键边界：新方案不加载旧权重，不把旧模型当 teacher，不复用旧共享主干、旧 Coslayer、旧输出头和旧损失组合。旧结果只作为失败分析证据和最终对照基线。
>
> 方案 A 的逐步数学推导、网络结构和训练目标见：[SCHEME_A_MATH_AND_TRAINING_GUIDE.md](./SCHEME_A_MATH_AND_TRAINING_GUIDE.md)。

## 0. 结论先行

当前训练困难不是单一的“loss 数值太大”，也不是模型宽度不足。它是以下问题叠加后的必然结果：

1. JFO 空化本身是互补约束控制的自由边界问题，不是一个全域光滑的普通椭圆 PDE。
2. 当前膜厚平滑宽度很小，`Lambda=604.0794` 又很大，使已知强迫项局部达到 `2.1632e4`。
3. 当前强形式需要对网络压力做二阶自动微分，并要求多个 `1e4` 量级项相消到 `1e1` 甚至更低。
4. Reynolds MSE、FB 互补损失和 `p*gamma` 损失相差许多数量级，而且三者的曲率、梯度方向和收敛速度也不同。
5. 旧 gamma 头的 `1e-6` 初始化和 `tanh^2` 让空化分支初期几乎没有梯度；旧压力头、可训练边界距离函数和共享主干又引入额外耦合。
6. 小批量 loss 选 best、Adam 长时间训练以及高梯度区域采样不足，会放大后期反弹和 checkpoint 偶然性。

因此，正确方向不是继续往旧模型上加一项 loss，而是同时重构三件事：

- **重构方程**：把二阶强形式改写为一阶通量本构关系，并用控制体积分守恒代替点态二阶残差。
- **重构约束**：把互补条件视为约束优化或直接编码为相结构，而不是固定权重的附加 MSE。
- **重构求解路径**：从简单物理问题连续推进到真实 `Lambda`、真实膜厚和锐利空化界面，而不是随机初始化后一次性硬解最终问题。

本文推荐优先实现：

| 方案 | 核心 | 是否复用旧模型 | 风险 | 推荐顺序 |
|---|---|---:|---:|---:|
| A：CV-AL PINN，可升级 MF-CV-AL | 一阶通量 + 控制体守恒 + 增广拉格朗日互补约束 | 否 | 中 | **第一优先** |
| B：FB-AS PINN | 显式水平集自由边界 + 满膜/空化专家 + 控制体守恒 | 否 | 高 | A 的物理基线稳定后 |

## 1. 当前问题的数学结构

### 1.1 控制方程

当前代码对应的无量纲控制方程为

```text
1/R * d/dR(R * H^3 * dp/dR)
+ 1/R^2 * d/dtheta(H^3 * dp/dtheta)
- Lambda * d/dtheta[(1-gamma) * H]
= 0
```

其中：

- `p` 是无量纲压力；
- `gamma` 是空化率或气相体积分数；
- `s = 1-gamma` 是液相饱和度；
- `H` 是无量纲膜厚；
- `Lambda = 604.0794`。

JFO 条件为

```text
p >= 0
0 <= gamma <= 1
p * gamma = 0
```

这意味着同一个计算域中存在两种不同状态：

```text
满膜区: p > 0, gamma = 0
空化区: p = 0, gamma > 0
界面区: p = 0, gamma = 0，并满足质量通量连接条件
```

所以 `gamma` 的连续数值并不是单纯的二分类概率。在空化区内部，它表示局部液相不足程度，数值大小具有物理意义；只有“空化区边界位置”这个评价任务才需要对 gamma 取阈值。

### 1.2 计算域与系数

```text
R in [47/52, 1] = [0.9038461538, 1]
theta in [0, 2*pi/6] = [0, 1.0471975512]
P_i = 0.0666667
P_o = 0.1
H approximately in [1, 4]
```

`H^3` 因而在约 `1` 到 `64` 之间变化。即使不考虑空化，扩散系数也已经具有 64 倍对比度；再叠加窄过渡带，强形式算子的条件数会明显恶化。

## 2. 旧模型的定量尸检

### 2.1 已达到的效果

两次已有完整评估的结果为：

| 指标 | SiLU warmup 结果 | tanh 结果 |
|---|---:|---:|
| `P_rel_L2` | `3.167989e-2` | `6.935375e-1` |
| `P_rel_Linf` | `2.018417e-1` | `9.795133e-1` |
| `gamma_rel_L2` | `1.607732e-1` | `7.365604e-1` |
| `gamma_rel_Linf` | `6.824110e-1` | `9.651188e-1` |
| `max(p*gamma)` | `3.480589e-3` | `8.574732e-3` |
| 单阈值 cavitation IoU | `0.304116` | `0.260288` |
| 单阈值 Dice | `0.466394` | `0.413061` |

SiLU 明显优于 tanh，说明激活函数和训练策略确实影响可训练性；但 SiLU 仍没有解决 formulation 的上限。压力主体和空化区大致位置已经出现，gamma 的幅值、界面位置和互补精度却仍明显不足，而且训练过程不稳定。普通残差连接实验也被实践观察为明显变差，因此它不再作为新项目默认结构。

### 2.2 为什么初始 Reynolds loss 自然达到 1e8

当前膜厚函数的 theta 平滑宽度是

```text
xi_theta = (2*pi/6) / 50
         = 0.02094395
```

槽深使膜厚跨越约 `Delta H=3`。单个 sigmoid 边缘的最大导数约为

```text
max |dH/dtheta| approximately Delta H / (4*xi_theta)
                        approximately 3 / (4*0.02094395)
                        approximately 35.81
```

因此已知驱动项的局部峰值约为

```text
Lambda * max|dH/dtheta|
approximately 604.0794 * 35.81
approximately 21632
```

这与诊断文件中的

```text
part_3_1 P99 = 21632.0
```

精确吻合。残差采用平方均值后，单点 `2.16e4` 会贡献约 `4.68e8`。所以初始 `loss_L_Reynolds = 1.0812e8` 不是意外，也不是简单的代码溢出，而是当前强形式、平滑宽度和物理参数共同决定的数值尺度。

### 2.3 真正危险的是大项相消

在约 22000 epoch 的较好快照中：

| 分项 | P99 |
|---|---:|
| 径向压力项 `part_1` | `2.1565e4` |
| 周向压力项 `part_2` | `2.5334e3` |
| 已知膜厚驱动 `part_3_1` | `2.1632e4` |
| gamma 输运项 `part_3_2` | `2.1829e4` |
| 合成残差 `f_p` | `3.2191e1` |

也就是说，网络必须让几个约 `20000` 的量相消，才能把残差压到约 `30`。以 P99 粗略估算，相消相对精度已经接近

```text
32 / 21632 approximately 1.5e-3
```

如果目标还要再下降多个数量级，网络导数、自动微分、采样和优化都必须保持极高一致性。对普通单精度、随机小批量、二阶网络导数的 PINN 来说，这不是合理的主训练目标。

### 2.4 单纯缩放 loss 为什么不够

假设把 PDE 残差整体乘 `1e-4`：

```text
r_new = 1e-4 * r_old
L_new = 1e-8 * L_old
```

这会让日志中的 `1e8` 变成约 `1`，并改善 PDE 与其他 loss 的表面量级平衡。但是：

- PDE 内部仍然是几个大项相消；
- 二阶微分算子没有改变；
- 单个 PDE loss Hessian 的所有特征值只被同一常数缩放，其内部条件数基本不变；
- 自由边界和互补约束没有改变；
- gamma 分支的死梯度和共享主干冲突没有改变。

所以缩放是必要的预条件化组成部分，但不是解决方案本身。有效的预条件化必须至少做到“不同方程分别无量纲化”，最好进一步改变微分阶数和守恒表达。

### 2.5 Reynolds、FB 与 p*gamma 并不是等价难度

22000 epoch 快照中：

```text
L_Reynolds = 7.3544e1
L_FB       = 5.9123e-7
```

两者相差约八个数量级。初始时 Reynolds 约 `1e8`，FB 约 `1e-17`，差距更大。

更重要的是，`p*gamma` 不是 FB 的真正替代品：

```text
d(p*gamma)^2/dp     = 2*p*gamma^2
d(p*gamma)^2/dgamma = 2*gamma*p^2
```

当任一变量已经很小时，另一变量收到的梯度也会很小。它能减少重叠，却不能可靠决定某点应该进入满膜相还是空化相。FB 也存在原点附近非光滑、固定罚因子难选和平凡相状态的问题。继续把二者一起加权，只会形成更多需要手调的竞争目标。

### 2.6 旧输出头给优化增加了额外障碍

旧架构的主要问题包括：

1. gamma 最后一层 kernel 初始化为 `1e-6`。
2. 旧 gamma 变换为 `tanh(sigma*g_raw)^2`，在零点附近导数也接近零。
3. 诊断中 epoch 0 的 gamma 均值、标准差和最大值全为零，验证了空化分支的初始死区。
4. 压力和 gamma 共用主干，gamma 深层输出头还拼接压力隐藏变量，两类相状态的梯度被强制写入同一表示。
5. 压力头使用可训练 Hermite 修正、可训练边界距离函数和 `tanh^2`，输出映射本身就具有饱和与多重参数耦合。
6. 旧 gate 使用 `x_t*U + (1-x_t)*V`，但 `x_t=tanh(...)` 位于 `[-1,1]`，它不是凸组合；当 `x_t<0` 时会放大 `V`，可能放大高阶导数。
7. 普通残差连接直接相加，不能保证 PINN 初始化时导数稳定。用户观察到它明显变差是可信的。

这解释了为什么“换 tanh”“开普通 residual”“继续加深输出头”没有触及真正瓶颈。

### 2.7 训练轨迹显示的是病态优化，不是正常收敛

诊断快照中的 Reynolds MSE：

```text
epoch 0      1.0812e8
epoch 7000   2.6730e2
epoch 13000  8.7015e3
epoch 22000  7.3544e1
epoch 24000  1.1720e3
epoch 27000  4.5780e3
epoch 29000  7.6324e3
```

它不是在低学习率下平稳收敛，而是在狭窄谷底附近反复逃逸。长时间 Adam 并没有可靠地把好状态保持住。

此外，旧 best 逻辑每 100 epoch 用当前训练 batch 的 total loss 判断并保存。即使评估脚本后来加载了 best checkpoint，这个 best 仍可能只是某个随机 batch 上的偶然最优，而不是固定全域网格上的物理最优。

### 2.8 根因优先级

| 优先级 | 根因 | 证据 | 仅加 loss 能否解决 |
|---:|---|---|---:|
| 1 | 二阶强形式 + 尖锐 H + 大 Lambda | `part_3_1 P99=21632`，多项强相消 | 否 |
| 2 | JFO 自由边界/互补结构 | p、gamma 属于不同活跃集，存在退化状态 | 否 |
| 3 | 微分算子导致病态 loss landscape | 大幅反弹、低残差需要极精确相消 | 否 |
| 4 | 多目标尺度与收敛速度失衡 | Reynolds 与 FB 相差 8 到 25 个数量级 | 部分 |
| 5 | 旧输出头与共享主干 | gamma 初始全零、复杂可训练边界映射 | 否 |
| 6 | 小批量 best 与单一 total loss | 最优状态和最终状态明显分离 | 否 |

## 3. 文献调查得到的可迁移结论

下面只列与当前问题有直接关系的方法，不把“新名词”本身当作改进。

### 3.1 JFO 空化 PINN 文献

1. [HL-nets: Physics-informed neural networks for hydrodynamic lubrication with cavitation](https://www.sciencedirect.com/science/article/pii/S0301679X2300659X) 使用压力与空化率双输出、FB 互补条件和多任务权重平衡。这证明当前路线有文献基础，但它仍属于强形式加权路线，不能解释或消除本项目由尖锐 H 引起的 `2.16e4` 强迫。
2. [Xi, Deng, Li: mass-conserving cavitation PINNs with soft and hard constraints](https://www.sciopen.com/article/10.1007/s40544-023-0791-1) 明确讨论非负约束、硬边界和平凡解；论文还需要限制平均空化率来排除退化解。
3. [该论文开放 PDF](https://file.sciopen.com/sciopen_public/1773645503533760513.pdf) 明确指出膜厚不连续时标准 PINN 难以直接处理，因为强形式依赖可微性。这一点与本项目为 H 人工设置窄 sigmoid 过渡完全对应。
4. [Physics-Informed Neural Networks for the Reynolds Equation with Transient Cavitation Modeling](https://www.mdpi.com/2075-4442/12/11/365) 同样表明满足 FB 并不自动意味着满足 Reynolds 方程，空化约束和守恒残差必须共同验证。

结论：不能把 FB 降低当作问题已经解决，也不能假定一个全域光滑 MLP 很适合跨越满膜和空化两种状态。

### 3.2 PINN 为什么难训练

1. [When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527) 证明不同 loss 分量具有显著不同的收敛速率，并讨论频谱偏置。
2. [Characterizing possible failure modes in PINNs, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html) 指出失败通常不是网络表达能力不足，而是微分算子使优化问题病态；课程式逐步增加 PDE 难度可降低误差。
3. [Challenges in Training PINNs: A Loss Landscape Perspective, ICML 2024](https://proceedings.mlr.press/v235/rathore24a.html) 从 Hessian 与微分算子角度解释病态性，并显示 Adam 后接 L-BFGS 比单独 Adam 更可靠。

结论：扩大网络或延长 Adam 训练不是优先方向。应降低微分阶数、做方程级预条件化、使用 continuation，并在后期使用准牛顿方法。

### 3.3 一阶混合形式、弱形式与控制体

1. [FO-PINN: A First-Order formulation for PINNs](https://www.sciencedirect.com/science/article/pii/S0955799725000499) 通过引入辅助变量把高阶 PDE 改成一阶系统，减少高阶自动微分并改善精度和速度。
2. [Mixed formulation PINNs for heterogeneous domains](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7388) 显示输出通量/梯度并结合一阶强弱形式，特别适合非均匀系数和材料跳变。
3. [Variational Physics-Informed Neural Networks](https://arxiv.org/abs/1912.00873) 通过分部积分降低对高阶导数的要求。
4. [Control-volume PINNs for conservation laws](https://www.sciencedirect.com/science/article/pii/S0021999121006495) 使用控制体积分守恒代替纯点态残差，在间断和守恒问题上更自然。

结论：本项目最值得改变的不是隐藏层，而是把 Reynolds 方程从二阶点态残差改成一阶通量本构 + 控制体质量守恒。

### 3.4 约束优化与硬约束

1. [Physics and Equality Constrained Artificial Neural Networks](https://arxiv.org/abs/2109.14860) 指出把 PDE、边界和其他条件全部塞进固定权重复合 loss 是严重限制，并使用增广拉格朗日法处理约束。
2. [AL-PINNs](https://arxiv.org/abs/2205.01059) 将增广拉格朗日用于 PINN 约束，避免只依赖固定罚权重。
3. [Exact imposition of boundary conditions in PINNs](https://www.sciencedirect.com/science/article/pii/S0045782521006186) 说明固定距离函数可以把边界条件从优化目标中移除。
4. [JFO Reynolds equation with an augmented-Lagrangian formulation](https://onlinelibrary.wiley.com/doi/full/10.1002/pamm.202300216) 表明增广拉格朗日在传统 JFO 数值求解中也有直接对应，而不是为了神经网络生造的技巧。

结论：互补关系应作为约束或相结构处理，不应继续依赖一个固定 `lambda_FB`。

### 3.5 架构与自由边界

1. [PirateNets, JMLR 2024](https://jmlr.org/papers/v25/24-0313.html) 指出普通深 MLP 的初始化会使网络导数难训练；其自适应残差块以 `alpha=0` 初始化，让网络从浅模型逐步变深。
2. [XPINNs](https://www.global-sci.com/cicp/article/view/6911) 和 [cPINNs](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127) 支持按物理区域分解网络，并通过界面残差或通量守恒连接。
3. [A free-boundary problem in lubrication theory](https://ddd.uab.cat/pub/pubmat/pubmat_a1989v33n2/pubmat_a1989v33n2p235.pdf) 从数学上把润滑空化视为自由边界问题。
4. [Adaptive finite elements for an inequality-constrained Reynolds problem](https://arxiv.org/abs/1711.04274) 表明变分不等式/鞍点表达能够直接针对空化边界，而不是靠事后阈值定义物理。

结论：用户已经验证“普通 residual 直接相加”效果差，这不等于所有自适应残差架构都无效；但新项目首个基线仍应使用简单独立网络，PirateNet 只作为严格对照实验。显式自由边界应作为第二方案，而不是一开始就增加实现风险。

## 4. 新项目共同设计原则

1. 从零初始化，不加载任何 `epochs_best_model` 或 SavedModel。
2. 不使用旧网络结构，不迁移旧 Coslayer、共享主干、Hermite 输出头或可训练 ADF。
3. 不使用 PCGrad、PIKAN、旧 RAD 和普通直接相加 residual。
4. theta 周期性由固定 Fourier 特征硬编码，不通过周期边界 loss 学习。
5. 径向压力边界和 gamma 径向边界尽可能由固定解析变换硬编码。
6. 不直接计算 `dH/dtheta`，不对压力做二阶自动微分。
7. 所有方程按固定物理特征尺度分别无量纲化；不使用当前 batch 的瞬时 loss 反向定义尺度。
8. 互补条件使用增广拉格朗日或相门控，不再把 `FB + p*gamma` 当作两个固定权重 MSE。
9. 训练采用物理 continuation：先简单、后真实；所有阶段都属于新模型自身训练，不是旧模型蒸馏。
10. best checkpoint 由固定验证网格/控制体上的多指标决定，不由某个随机训练 batch 决定。

## 5. 方案 A：CV-AL PINN，可升级 MF-CV-AL

最小版本全称：**Control-Volume Augmented-Lagrangian PINN**。加入独立 FluxNet 后升级为 **Mixed-Flux Control-Volume Augmented-Lagrangian PINN**。

这是推荐首先实现的方案。它保留 PINN 的核心：神经网络表示未知场，训练信号来自控制方程和约束；但它不再使用旧二阶强形式。

### 5.1 把 Reynolds 方程改写为守恒通量

原方程乘以 `R`：

```text
d/dR(R H^3 p_R)
+ d/dtheta[(H^3/R) p_theta - Lambda R (1-gamma)H]
= 0
```

定义总通量

```text
q_R     = R H^3 * p_R
q_theta = (H^3/R) * p_theta - Lambda R (1-gamma)H
```

得到三个一阶关系：

```text
r_R     = q_R - R H^3 p_R = 0
r_theta = q_theta - (H^3/R)p_theta + Lambda R(1-gamma)H = 0
r_mass  = d(q_R)/dR + d(q_theta)/dtheta = 0
```

但主方案不直接对 `r_mass` 做点态微分，而是在每个控制体 `Omega_c` 上使用

```text
integral_boundary(Omega_c) q dot n ds = 0
```

对矩形控制体可写成

```text
int_theta [q_R(R_right,theta)-q_R(R_left,theta)] dtheta
+ int_R [q_theta(R,theta_top)-q_theta(R,theta_bottom)] dR
= 0
```

这样做的直接收益：

- 压力只需一阶导数；
- 控制体守恒不需要对 q 再求导；A-min 只在控制体边界计算 p 的一阶导数，A-mixed 的守恒项本身不需要网络导数；
- 不再显式出现 `dH/dtheta`；
- 膜厚可以逐步从平滑 H 过渡到更锐利甚至分片 H；
- 训练目标直接约束净质量通量，而不是依赖几个巨大局部项相消。

### 5.2 新网络不是“双输出旧 MLP”

方案 A 分为两级。首个最小基线只使用两个彼此独立的网络：

```text
PressureNet: (R, theta, geometry features) -> z_p
GammaNet:    (R, theta, geometry features) -> z_gamma
```

通量直接由 `p`、`gamma` 和一阶压力导数按物理公式计算，再进入控制体积分。两网不共享隐藏主干，只通过物理通量和互补约束耦合。

若最小基线证实压力一阶导数噪声、跨槽通量连续性或尖锐 H 下的通量表示成为主要瓶颈，再升级到完整混合形式，新增：

```text
FluxNet: (R, theta, geometry features) -> z_qR, z_qTheta
```

此时 FluxNet 由两个本构一致性残差约束，不能独立编造通量。这样可以先验证控制体 formulation，再判断第三个网络是否真正必要。

首个基线建议：

```text
fixed Fourier embedding
4 hidden layers per network
width 128
SiLU activation
float64
no ordinary residual skip
```

输入特征：

```text
rho = normalized R
sin(n*K*theta), cos(n*K*theta), n=1..N_fourier
H(R,theta)
optional normalized signed distances to groove edges
```

theta 不直接以非周期标量输入，确保网络输出天然周期。

### 5.3 输出约束

令

```text
t = (R-R_i)/(R_o-R_i)
d_R = 4*t*(1-t), in [0,1]
softplus_beta(x) = log(1+exp(beta*x))/beta
```

压力可写为

```text
a_bc(t) = (1-t)*softplus_beta_inverse(P_i)
          + t*softplus_beta_inverse(P_o)
p = softplus_beta(a_bc + d_R*z_p)
```

它具有：

- 径向边界严格满足 `P_i/P_o`；
- 压力非负；
- 内部可以通过负 latent 接近零；
- 不使用 `tanh^2`，避免额外饱和。

gamma 可写为

```text
gamma = d_R * sigmoid(z_gamma)
```

从而满足 `0<=gamma<=1`，并在径向压力边界保持满膜。如果实际物理边界并不要求两侧 `gamma=0`，只需要更换固定边界变换，不要重新引入边界 MSE。

仅在加入 FluxNet 的 A-mixed 版本中，通量使用固定尺度输出：

```text
q_R     = S_qR     * z_qR
q_theta = S_qTheta * z_qTheta
```

### 5.4 固定方程尺度

选取固定参考量：

```text
Delta_R     = R_o-R_i
Delta_theta = 2*pi/6
P_ref       = max(P_i,P_o) or another declared physical pressure scale
R_ref       = mean(R_i,R_o)
H_ref       = declared RMS or median H from geometry
```

通量尺度可取

```text
S_qR = R_ref * H_ref^3 * P_ref / Delta_R

S_qTheta = H_ref^3 * P_ref / (R_ref*Delta_theta)
           + Lambda*R_ref*H_ref
```

然后分别定义

```text
rhat_R     = r_R / S_qR
rhat_theta = r_theta / S_qTheta
```

控制体净通量按该控制体的特征边长归一化：

```text
S_cv = S_qR*Delta_theta_cell + S_qTheta*Delta_R_cell
rhat_cv = net_flux / S_cv
```

这些尺度在训练前由物理参数计算并写入配置和日志。不能每个 batch 改动，否则优化器面对的目标也在移动。

### 5.5 物理目标

两网最小版本直接按公式计算通量，因此物理目标只有控制体守恒：

```text
J_phys_Amin = L_CV = mean(rhat_cv^2)
```

加入 FluxNet 后才增加本构一致性残差：

```text
L_const_R     = mean(rhat_R^2)
L_const_theta = mean(rhat_theta^2)
L_CV          = mean(rhat_cv^2)

J_phys_Amixed = L_const_R + L_const_theta + L_CV
```

混合版本的三项都是无量纲、预期约为 O(1) 的方程误差。它们不是旧 Reynolds 大残差与微小 FB 的直接相加。

控制体边界积分使用固定 Gauss-Legendre 点。首个实现建议每条边 3 或 4 个积分点，并对跨膜厚边缘的控制体额外细分。

### 5.6 互补条件使用增广拉格朗日

先定义尺度化变量

```text
p_bar = p/P_ref
g_bar = gamma
```

使用平滑 FB 约束

```text
c_FB = p_bar + g_bar
       - sqrt(p_bar^2 + g_bar^2 + epsilon_FB^2)
```

不再写成固定 `lambda_FB * mean(c_FB^2)`，而是采用外层增广拉格朗日：

```text
J_AL = J_phys
       + mean(lambda_i*c_FB_i)
       + mu/2 * mean(c_FB_i^2)
```

每个外层周期后：

```text
lambda_i <- lambda_i + mu*c_FB_i
```

仅当约束下降停滞时才增大 `mu`。固定一组 complementarity points，才能稳定维护点对点乘子。这样做的意义是：

- 乘子负责确定约束方向；
- `mu` 不需要一开始就极大；
- 不必让一个固定罚权重跨越整个训练过程；
- `p*gamma` 不再作为默认 loss；它只保留为独立评估指标。

`epsilon_FB` 使用 continuation，例如从 `1e-2` 逐步降到 `1e-5`。不要一开始就把非光滑原点直接交给随机初始化网络。

### 5.7 物理 continuation，而不是旧模型 warm start

推荐训练路径：

#### A0：算子单元测试

- 用制造解验证 `q_R/q_theta` 定义、控制体方向和周期边界。
- 用解析常数 H 情况验证一阶本构残差。
- float64 下检查自动微分的一阶导数。

#### A1：满膜低难度问题

```text
gamma fixed to 0
Lambda = 0.1*Lambda_true
H transition width = 4*xi_true
```

只训练 `PressureNet`，直接计算通量，使控制体守恒稳定。此阶段 GammaNet 固定输出零，不加入 FluxNet。

#### A2：提高驱动

```text
Lambda ratio: 0.1 -> 0.25 -> 0.5 -> 0.75 -> 1.0
```

每一级都从同一套新网络上一阶段参数继续，不加载旧项目权重。从约 `0.25*Lambda_true` 开始开放 GammaNet 和平滑互补约束，避免一直强迫真实空化问题保持满膜。

#### A3：激活空化

- 解冻 `GammaNet`；
- 启用平滑 FB 增广拉格朗日；
- `epsilon_FB: 1e-2 -> 1e-3 -> 1e-4 -> 1e-5`；
- 每个 epsilon 阶段重新评估 active set 和全域守恒。

#### A4：锐化真实膜厚

```text
xi_H ratio: 4 -> 2 -> 1
```

若控制体方案在 `xi=1` 已稳定，再做 `0.5` 或分片 H 实验。分片 H 不是首个验收条件。

#### A5：准牛顿收敛

- Adam 负责进入合理盆地；
- 固定采样集后使用 full-batch L-BFGS；
- 每轮 AL 乘子更新后可短暂回到 Adam，再进入 L-BFGS；
- 不再用 30000 次纯 Adam 当作默认成功路径。

#### A6：有证据时升级 FluxNet

- 如果 A-min 已稳定且精度满足要求，停止升级；
- 如果压力一阶导数噪声或跨槽通量成为明确瓶颈，新增 FluxNet；
- 加入两个归一化本构一致性残差；
- 将 A-min 与 A-mixed 作为严格消融，不默认后者更好。

### 5.8 采样与控制体

固定、可复现地分层：

```text
40% uniform control volumes
25% groove/H transition neighborhoods
20% current complementarity transition band
10% radial boundary neighborhoods
 5% periodic seam checks
```

其中“当前 transition band”每个 AL 外层周期更新一次，不在每个 batch 随机重采样。控制体不能跨越周期 seam 后丢失对侧通量；theta 方向应按周期索引连接。

### 5.9 best checkpoint

固定验证集上保存以下独立量：

```text
E_const_R
E_const_theta
E_CV_mean
E_CV_p99
E_CV_global_net_flux
E_FB_mean
E_FB_p99
max(p*gamma)
periodic_value_error
periodic_flux_error
```

best 采用可行性优先的字典序：

1. 边界和全局质量守恒必须达标；
2. 比较 `E_CV_p99`；
3. 比较 `E_FB_p99`；
4. 再比较平均本构残差。

FEM 的 L2 和 IoU 只作为实验评估，不参与训练或 checkpoint 选择，确保新 PINN 真正从物理方程出发。

### 5.10 为什么方案 A 直接针对旧根因

| 旧问题 | 方案 A 的处理 |
|---|---|
| 压力二阶自动微分 | A-min 改成一阶 p 导数；必要时再加入独立通量 |
| 显式 `dH/dtheta` 产生 21632 强迫 | 控制体与本构只读取 H 值，不直接微分 H |
| 多个大项点态相消 | 直接计算/拟合通量并约束控制体净通量 |
| FB 权重难选 | 增广拉格朗日外层更新 |
| p/gamma 共享主干冲突 | 三个独立网络 |
| tanh-square 死梯度 | softplus/sigmoid 固定变换 |
| 普通 residual 变差 | 首个基线完全不使用 residual |
| Adam 后期反弹 | continuation + full-batch L-BFGS |
| 随机 batch best | 固定控制体多指标验证 |

## 6. 方案 B：FB-AS PINN

全称：**Free-Boundary Active-Set PINN**。

方案 B 更激进：不让一个平滑 gamma 头自己隐式发现相区，而是显式学习空化自由边界，并让满膜区和空化区使用不同专家。

### 6.1 网络组成

```text
InterfaceNet: phi(R,theta)          # phi>0 满膜，phi<0 空化
PressureExpert: p_plus(R,theta)     # 满膜压力候选
GammaExpert: gamma_plus(R,theta)    # 空化率候选
FluxNet: q_R(R,theta), q_theta(R,theta)
```

定义软 active-set 门：

```text
m_tau = sigmoid(phi/tau)
```

并使径向边界保持满膜：

```text
m = 1-d_R + d_R*m_tau
p = m*p_plus
gamma = d_R*(1-m)*sigmoid(z_gamma)
```

这样在 `tau -> 0` 时：

```text
满膜区 m=1: p=p_plus, gamma=0
空化区 m=0: p=0, gamma=gamma_plus
```

互补关系主要由模型结构实现，而不是由全域 FB MSE 强迫。

### 6.2 界面条件

在 `phi=0` 附近使用平滑 delta 带

```text
delta_tau(phi) = sigmoid(phi/tau)*(1-sigmoid(phi/tau))/tau
```

约束：

```text
L_interface_p = weighted mean of p_plus^2 near phi=0
L_interface_flux = normal flux jump across phi=0
L_eikonal = mean((|grad phi|-1)^2) near interface
```

`L_interface_p` 确保空化边界压力趋近零；`L_interface_flux` 保持质量连接；`L_eikonal` 让 phi 接近符号距离，防止水平集梯度任意缩放。

这些界面条件也建议使用增广拉格朗日，而不是极大的固定罚权重。

### 6.3 区域物理

仍使用方案 A 的混合通量和控制体守恒。区别是本构关系按 active set 加权：

```text
full-film constitutive residual weighted by m
cavitation transport residual weighted by 1-m
control-volume total mass residual enforced everywhere
```

跨界面的控制体尤其重要，因为它们直接约束界面两侧的净质量流量，不依赖 `phi=0` 的点采样恰好落在哪里。

### 6.4 训练顺序

#### B0：无空化初始化

- `phi` 初始化为全域正值；
- `gamma=0`；
- 从低 Lambda、宽 H 开始训练压力和通量。

#### B1：允许界面成核

- 在压力接近零且质量残差持续高的区域释放 InterfaceNet；
- `tau` 取较大值，例如 `0.1`；
- 每次只更新少量界面参数，避免全域突然塌缩为空化相。

#### B2：交替 active-set

```text
freeze phi -> optimize pressure/gamma/flux
freeze experts -> optimize phi and interface constraints
update AL multipliers
```

不建议四个网络从第一步开始完全联合自由训练。

#### B3：锐化

```text
tau: 0.1 -> 0.05 -> 0.02 -> 0.01
```

每次降 tau 前必须满足全局质量守恒和界面通量阈值。锐化不是为了提高视觉效果，而是为了让 active set 收敛。

### 6.5 方案 B 的优势和风险

优势：

- 空化边界成为模型直接输出，不依赖单个 gamma 阈值；
- 满膜与空化两种 PDE 状态不再挤在一个平滑输出头中；
- 互补条件在 `tau -> 0` 时由结构满足；
- 可以分别统计界面位置误差、区域内 gamma 幅值误差和通量误差。

风险：

- `phi`、p、gamma、q 的交替优化更复杂；
- 可能出现全满膜或全空化的相塌缩；
- 多个空化区的生成、合并和拓扑变化需要足够 Fourier 表达能力；
- 界面条件若写错，模型可能拥有漂亮边界但不守恒；
- 实现和调试成本大约是方案 A 的 1.5 到 2 倍。

所以 B 不应取代 A 的最小物理基线，而应在 A 验证“一阶通量 + CV 守恒”有效后再进入。

## 7. 两套方案对比

| 项目 | 方案 A：CV-AL / MF-CV-AL | 方案 B：FB-AS |
|---|---|---|
| 相区表示 | p、gamma + AL 互补 | 显式 phi active set |
| 压力最高微分阶数 | 一阶 | 一阶 |
| H 的显式导数 | 无 | 无 |
| 质量守恒 | 控制体 | 控制体 |
| 互补处理 | 增广拉格朗日 FB | 结构门控 + 界面约束 |
| 是否需要 gamma 阈值定义边界 | 评估时需要多阈值 | 不需要，直接用 phi=0 |
| 网络数 | 首版 2，混合升级 3 | 4 |
| 训练复杂度 | 中 | 高 |
| 首次成功概率 | 较高 | 中 |
| 界面上限 | 中高 | 高 |
| 推荐 | 首先实现 | 第二阶段研究 |

## 8. 必须执行的最小消融实验

不能再一次训练一个庞大组合，然后无法判断哪一项有效。方案 A 至少做以下顺序：

| 实验 | 方程 | 约束 | 目的 |
|---|---|---|---|
| A-00 | 常数 H 制造解 | 无空化 | 验证通量符号和 CV 积分 |
| A-01 | 真实 H、低 Lambda | gamma=0 | 验证 H 过渡下的一阶形式 |
| A-02 | 真实 Lambda | gamma=0 | 确认满膜问题是否可稳定训练 |
| A-03 | 真实 Lambda | 固定罚权重 FB | 只作为反例基线 |
| A-04 | 真实 Lambda | AL-FB | 验证约束优化收益 |
| A-05 | A-04 | Adam + L-BFGS | 验证后期预条件化 |
| A-06 | A-05 | H continuation | 验证锐利几何稳定性 |
| A-07 | A-06 | adaptive residual block | 只验证 PirateNet 是否值得保留 |

每个实验至少运行 3 个随机种子，报告均值、标准差和最差种子。单次最好结果不能再作为方案有效的证据。

## 9. 成功标准

### 9.1 物理可行性先于 FEM 误差

新模型必须先满足：

```text
radial pressure BC error <= 1e-8 in float64
periodic value/flux error <= 1e-8 when hard encoded
global normalized net mass flux <= 1e-5
CV residual p99 <= declared threshold
max(p*gamma) <= declared threshold
no NaN/Inf in p, gamma, q or gradients
```

阈值可在制造解测试后调整，但必须在正式训练前固定。

### 9.2 与 FEM 的后验评价

```text
P relative L2 and Linf
gamma relative L2 and Linf
pressure error in full-film region
gamma error inside cavitation region
interface distance error
global load and mass-flow error
```

IoU 必须报告多阈值，而不是单个 `1e-6`：

```text
gamma threshold = 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
```

同时报告 IoU 对阈值的曲线或表格。如果 IoU 对阈值极敏感，说明模型主要问题是非空化区存在 gamma 尾值；如果所有阈值都低，才说明空化边界位置本身错误。

方案 B 还要直接报告 `phi=0` 与 FEM 空化边界的 Chamfer/Hausdorff 距离，不再用 gamma 阈值替代界面。

### 9.3 目标不应写成“raw loss 必须到 1e-6”

不同残差经过不同固定尺度和积分后，raw 数值不再与旧 loss 可比。真正目标是：

- 各无量纲方程残差达到预先声明的容差；
- 全局质量平衡成立；
- 互补/active set 可行；
- FEM 场误差和界面误差改善；
- 多随机种子训练稳定。

追求旧 raw Reynolds MSE 的 `1e-6` 没有物理必要性，也可能在数值上不现实。

## 10. 新项目建议目录

```text
0504_next/
  configs/
    mf_cv_al_base.py
    free_boundary_as.py
  geometry/
    film_thickness.py
    periodic_features.py
  models/
    pressure_net.py
    gamma_net.py
    flux_net.py
    interface_net.py
    output_transforms.py
  physics/
    mixed_flux.py
    control_volume.py
    complementarity.py
    augmented_lagrangian.py
  training/
    continuation.py
    adam_stage.py
    lbfgs_stage.py
    checkpoint.py
  evaluation/
    physics_metrics.py
    fem_metrics.py
    interface_metrics.py
    multithreshold_iou.py
  tests/
    test_manufactured_solution.py
    test_flux_signs.py
    test_control_volume.py
    test_hard_boundaries.py
    test_periodicity.py
```

这个目录与旧 `tensordiffeq` 解耦。可以读取相同物理参数和 FEM 文件，但不 import 旧网络、旧 optimizer、旧 fit 或旧 loss 实现。

## 11. 明确停止的方向

以下方向不再进入主实验：

1. 在旧强形式上继续增加 `p*gamma`、active-set MSE 或更多固定权重项。
2. 只把 Reynolds loss 乘一个常数后宣称病态性已经解决。
3. 继续扩大旧共享 MLP 或输出头深度。
4. 继续尝试普通直接相加 residual。
5. 恢复 PCGrad 或 PIKAN。
6. 用旧 SiLU 模型做 teacher、伪标签或初始化。
7. 用 FEM 场参与训练来掩盖物理 loss 失败。
8. 用单次 seed、单个 IoU 阈值或 final epoch 判断成功。
9. 让随机 mini-batch total loss 决定 best checkpoint。
10. 在没有制造解和通量符号测试前直接运行数万 epoch。

## 12. 最终推荐

先实现方案 A，而且第一个里程碑不是“在 FEM 上超过旧模型”，而是：

```text
不显式求 dH/dtheta
不求 p 的二阶导数
在真实几何上稳定满足控制体质量守恒
训练轨迹不再从 1e2 反弹到 1e4
AL 约束可以逐步降低互补误差
```

只有这五件事成立，才说明新 formulation 真正移除了旧模型的核心瓶颈。随后再看压力、gamma 和空化边界是否超过旧基线。

若方案 A 的压力和质量守恒稳定，但 gamma 界面仍然宽、IoU 对阈值高度敏感，再进入方案 B。那时可以判断瓶颈已经从“PDE 病态训练”转移到“自由边界表示”，显式 InterfaceNet 才是有依据的下一步，而不是新的盲目堆叠。

**一句话决策：旧项目的问题不在于少了哪个 loss，而在于用一个全域光滑共享网络，通过二阶点态强残差和固定罚权重，硬解一个尖锐系数、强相消、互补自由边界问题。新项目应从守恒通量和约束结构重新开始。**
