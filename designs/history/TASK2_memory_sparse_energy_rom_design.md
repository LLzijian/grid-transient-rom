# 任务二：记忆闭合、机械功率动力学与稀疏能量结构的低阶模型设计

## 1. 总体目标

基于任务一构建的私有 post-fault 数据集，建立一个从部分可观测高阶电力系统暂态中发现**低阶、显式、稀疏并可进行稳定性分析的动力学模型**。

研究主线为：

\[
\boxed{
\text{short history}
\rightarrow
\text{minimal hidden state}
\rightarrow
\text{explicit sparse ODE}
\rightarrow
\text{energy / stability structure}
}
\]

最终模型应同时满足：

1. 使用短时 \(\delta,\omega\) 历史恢复当前无法由单点确定的隐藏状态；
2. 显式建模 \(P_m\) 动态，不长期采用 ZOH；
3. 动力学方程本身保持显式、低阶、稀疏和物理受约束；
4. 黑盒仅用于状态估计，不直接生成长期未来轨迹；
5. 模型可计算平衡点、Jacobian、局部模态、能量函数及稳定裕度；
6. 只增加对预测、结构或稳定性确有必要的 reduced states。

---

## 2. 核心建模假设

真实 post-fault 系统为高阶自治系统：

\[
\dot x=F(x)
\]

可观测量主要为：

\[
y=[\delta,\omega]
\]

低阶建模不要求 \((\delta,\omega)\) 的单一瞬时值独立决定未来，而是使用一段短时历史：

\[
\mathcal H_t=\{\delta,\omega\}_{t-T_{obs}:t}
\]

从历史中恢复一个低维 memory state：

\[
q_e(t)=E_\phi(\mathcal H_t)
\]

随后所有未来演化由显式 reduced ODE 完成。

\(P_m\) 作为物理意义明确、可直接观测的机械侧动态变量，优先于 latent state 建模。只有在显式 \(P_m\) 动力学仍不足以闭合系统时，再引入 \(q_e\) 表示剩余 unresolved electrical/control memory。

---

## 3. 总体模型结构

目标 reduced state 为：

\[
\boxed{x_r=[\delta,\omega,P_m,q_e]}
\]

其中：

- \(\delta\)：转子角低阶状态；
- \(\omega\)：转速低阶状态；
- \(P_m\)：有效机械功率状态；
- \(q_e\)：剩余高阶电气/控制动态的低维 memory state。

最终目标形式：

\[
\dot\delta=\omega
\]

\[
M\dot\omega=P_m-\hat P_e(\delta,q_e)-D\omega
\]

\[
\dot P_m=f_m(P_m,\omega)
\]

\[
\dot q_e=f_e(\delta,\omega,P_m,q_e)
\]

其中 \(f_m\)、\(f_e\)、\(\hat P_e\) 优先通过物理约束与稀疏结构发现获得。

History encoder 只用于初始化：

\[
q_{e,0}=E_\phi(\mathcal H_{0:T_{obs}})
\]

第一阶段直接使用真实 \(P_{m,0}\)，不要求 encoder 同时估计 \(P_m\)。

---

## 4. 记忆模块设计

### 4.1 功能边界

Memory module 只负责：

> 根据最近一小段 post-fault \(\delta,\omega\) 轨迹估计当前 reduced hidden state \(q_e\)。

不允许：

- GRU/TCN 直接预测后续轨迹；
- encoder 在 rollout 过程中持续替代动力学；
- 使用 fault line、fault duration、fault type 等 metadata；
- 使用未来样本；
- 通过大模型容量掩盖显式 ODE 的结构缺陷。

### 4.2 输入窗口

\[
T_{obs}=50\text{ ms}
\]

### 4.3 Encoder 结构

优先顺序：

1. compact trend / delay features；
2. small GRU；
3. small TCN。

首选 hidden dimension：8 或 16。

输出维度首先尝试：

\[
\dim(q_e)=1
\]

仅在一维 memory state 无法满足跨扰动预测和稳定性要求时扩展到 2。

---

## 5. \(P_m\) 动力学设计

\(P_m\) 是算法的第一项新增物理状态，应先于 latent \(q_e\) 完成。

### 5.1 基础模型

第一版采用一阶 effective governor/turbine model：

\[
\boxed{
T_m\dot P_m=P_{ref}-P_m-K_\omega\omega
}
\]

辨识：

\[
T_m,\quad K_\omega,\quad P_{ref}
\]

并约束：

\[
T_m>0
\]

参数应具有稳定、合理的数量级。

### 5.2 稀疏修正

若一阶模型在多秒 horizon 上存在明确系统性残差，再增加少量候选项：

\[
T_m\dot P_m=P_{ref}-P_m-K_\omega\omega+
\Theta_m(P_m,\omega)\xi_m
\]

只允许少量 physically plausible terms，并通过 sparsity threshold 保留最小模型。

### 5.3 物理角色

\(P_m\)-ODE 负责吸收明确的机械侧 governor/turbine 慢动态。

在完成该模块前不引入 \(q_e\)，避免 latent state 同时承担机械侧和电气侧的混合记忆。

---

## 6. 稀疏能量函数发现

稀疏能量函数是最终模型的核心结构目标，但应建立在 M4 动力学已经基本可用之后。

### 6.1 优先发现低复杂度 storage / energy function

候选形式：

\[
\boxed{H(x_r)=\Theta_H(x_r)\xi_H}
\]

候选项优先包括：

- \(\omega^2\)；
- \(1-\cos(\delta_i-\delta_j)\)；
- \((P_m-P_m^*)^2\)；
- \(q_e^2\)；
- 必要的 \(q_e\sin\delta\) 或低阶 cross terms。

函数库保持小规模、物理先验驱动，不使用大规模任意高阶多项式库。

### 6.2 基本能量约束

至少要求：

1. 平衡点附近 \(H\) 有局部极小；
2. stable trajectories 上 \(H\) 总体呈耗散趋势；
3. \(\dot H\) 的大范围正增长受到约束；
4. energy structure 不显著损害 rollout accuracy。

### 6.3 更强结构

若稀疏 \(H\) 能稳定识别，再进一步尝试：

\[
\dot x_r=[J(x_r)-R(x_r)]\nabla H(x_r)
\]

其中：

\[
J=-J^T,\qquad R=R^T\succeq0
\]

完整 \(J/R\) 分解不是前置要求；只有在数据和已发现 ODE 支持时再推进。

---

## 7. 最小物理模型族

执行阶段只维护三类正式模型。

### M2：clean baseline

\[
[\delta,\omega]
\]

用途仅为统一比较基准。建立后冻结，不继续围绕 M2 做结构优化。

### M3-Pm

\[
\boxed{[\delta,\omega,P_m]}
\]

加入显式 \(P_m\)-ODE，回答机械功率动态能够解释多少长期误差。

### M4

\[
\boxed{[\delta,\omega,P_m,q_e]}
\]

在 M3-Pm 基础上加入 short-history observer 和最小 memory state，形成主要目标模型。

只有 M4 在明确指标上仍不足时，才允许加入第二个 memory state。

不单独维护 M3-q 作为正式开发分支。

---

## 8. 稀疏候选函数库

候选库必须围绕电力系统结构构造。

### 8.1 摆动/电气侧

优先：

\[
\sin(\delta_i-\delta_j),\qquad
\cos(\delta_i-\delta_j),\qquad
\omega_i
\]

以及与 \(q_e\) 的少量低阶耦合。

### 8.2 机械侧

优先：

\[
P_m-P_m^*,\qquad \omega
\]

必要时加入一两个 cross terms，而不是开放高阶多项式库。

### 8.3 memory state

首选：

\[
q_e,\qquad \omega,\qquad \sin\delta,\qquad P_m-P_m^*
\]

目标是得到少数可以解释的 closure terms。

---

## 9. Memory state 的物理解释策略

\(q_e\) 首先定义为：

> unresolved electrical/control memory state。

不要求它对应某一个单独的 ANDES state。

M4 训练完成后，用任务一保存的 full-order states 检查 \(q_e\) 与以下变量的关系：

- transient EMF；
- AVR states；
- internal voltage；
- \(P_e\)；
- generator terminal voltage；
- 其他高阶 electrical/control states。

使用：

- correlation；
- linear projection；
- PCA/subspace alignment；
- simple regression。

若 \(q_e\) 与某个物理组合高度一致，可进一步解释；否则保持 effective memory state 的定义。

---

## 10. 训练流程

整个开发按四个 milestone 顺序推进。每个 milestone 通过验收后建立 Git checkpoint，再进入下一阶段。

### Milestone 0：M2 clean baseline

在新数据上重建统一 M2 baseline，只完成：

- 数据 adapter；
- 统一 rollout；
- 多时域指标；
- train/val/test gate。

M2 建立后冻结，不继续寻找更复杂二阶修正。

### Milestone 1：M3-Pm

先使用真实 \(P_m(t)\) 识别 effective \(P_m\)-ODE。

完成后将预测的 \(P_m(t)\) 接入 swing dynamics，比较：

\[
\text{ZOH }P_m
\quad vs \quad
\text{dynamic }P_m
\]

重点评估 1–8 s horizon。

M3-Pm 完成可信辨识和贡献评估后进入 M4。若收益有限，应记录其解释能力和残余误差。

### Milestone 2：M4 memory closure

在已存在 \(P_m\)-ODE 的前提下加入：

\[
q_{e,0}=E_\phi(\mathcal H)
\]

并发现最简单的：

\[
\dot q_e=f_e(\delta,\omega,P_m,q_e)
\]

开发顺序：

1. 先用 compact history features 验证收益；
2. 再使用 small GRU/TCN 替代手工 history feature；
3. 固定 \(\dim(q_e)=1\) 完成第一版；
4. 只有一维确实不足才扩展到 2。

此阶段核心问题是：

> 在机械侧动态已显式建模后，一个最小 memory state 能否闭合剩余电气/控制记忆并改善跨扰动 rollout？

### Milestone 3：稀疏能量与稳定性结构

以通过验收的 M4 为基础：

1. 稀疏化 \(f_e\)、\(\hat P_e\) 及必要的 \(P_m\) residual；
2. 构建稀疏候选 energy/storage function \(H(x_r)\)；
3. 检查 equilibrium、\(\dot H\)、耗散与轨迹一致性；
4. 计算 Jacobian、局部模态和能量稳定裕度；
5. 必要时再尝试 \(J-R\) 结构化表达。

不在 M4 尚未证明有效时提前构建完整 port-Hamiltonian framework。

---

## 11. 损失函数

不同 milestone 使用尽量少的损失项。

### 11.1 M2 / M3-Pm

主要：

\[
\mathcal L_{traj}
\]

和：

\[
\mathcal L_{Pm}=\|\hat P_m-P_m\|^2
\]

### 11.2 M4

采用：

\[
\mathcal L=
\lambda_{traj}\mathcal L_{traj}
+\lambda_{phys}\mathcal L_{swing}
+\lambda_s\mathcal L_{sparse}
+\lambda_{tail}\mathcal L_{tail}
\]

其中：

\[
\mathcal L_{swing}=\|M\dot\omega-(P_m-P_e-D\omega)\|^2
\]

### 11.3 能量阶段

M4 已稳定后再加入：

\[
\mathcal L_H
\]

惩罚明显违反耗散结构的行为，例如：

\[
\mathrm{ReLU}(\dot H)^2
\]

避免从第一轮训练开始同时优化过多目标。

### 11.4 Tail-aware training

所有正式模型必须采用 event-balanced weighting，并监控：

- worst-k event loss；
- worst-10% CVaR；
- 不同 disturbance group 的最差表现。

平均指标改善不能以明显牺牲少数困难事件为代价。

---

## 12. 稳定性分析

稳定性分析只针对最终显式 reduced ODE：

\[
\dot x_r=F_r(x_r)
\]

History encoder 不参与 future dynamics，因此不妨碍对 \(F_r\) 的稳定性分析。

### 12.1 平衡点

求解：

\[
F_r(x^*)=0
\]

并验证与 post-fault equilibrium 一致。

### 12.2 局部稳定性

计算：

\[
A=\left.\frac{\partial F_r}{\partial x}\right|_{x^*}
\]

至少输出：

- dominant eigenvalue；
- damping ratio；
- modal frequency；
- parameter sensitivity。

必要时与 full-order 主导模态进行趋势比较。

### 12.3 非线性稳定性

优先使用发现的 \(H(x_r)\) 分析：

- equilibrium 是否为局部 minimum；
- stable trajectories 是否耗散；
- unstable trajectories 是否跨越 candidate energy boundary。

数值 basin/ROA 扫描用于验证 energy boundary，而不是先建立复杂 ROA framework。

---

## 13. 稳定裕度设计

优先使用与能量和经典稳定性概念相容的指标。

### 13.1 Energy margin

若存在可靠的 critical energy：

\[
\boxed{
\Delta H=H_{crit}-H(x_r)
}
\]

### 13.2 Normalized energy margin

\[
\eta_H=\frac{H_{crit}-H(x_r)}{H_{crit}}
\]

### 13.3 Local spectral margin

平衡点附近使用：

\[
\eta_\lambda=-\max_i\Re(\lambda_i)
\]

### 13.4 Numerical boundary distance

若需要，可在 reduced-state 空间数值重建局部稳定边界，并计算：

\[
d(x_r,\partial\mathcal A)
\]

该指标作为验证和补充，不替代能量裕度主线。

---

## 14. 黑盒与稳定性分析的边界

允许：

\[
q_{e,0}=E_\phi(\mathcal H)
\]

最终 future dynamics 必须为：

\[
\dot x_r=F_{sparse}(x_r)
\]

不采用：

\[
\dot x_r=NN(x_r)
\]

作为最终论文模型。

原则：

> **可以用黑盒恢复当前状态，不用黑盒替代动力学结构。**

这样 history encoder 是 observer，稳定性、能量和安全裕度分析都建立在显式 ODE 上。

---

## 15. 稳定域的研究对象

M4 的严格 reduced state 为：

\[
x_r=[\delta,\omega,P_m,q_e]
\]

因此严格稳定域为：

\[
\mathcal A\subset\mathbb R^{\dim(x_r)}
\]

二维 \(\delta-\omega\) 图仅用于：

- 固定 \(P_m,q_e\) 的切片；
- 对 reduced state 稳定结果的投影；
- 工程可视化。

模型不强制把高维稳定性压成一个天然唯一的二维硬边界。

---

## 16. 与物理结构发现目标的对应

传统 SINDy：

\[
\dot x=\Theta(x)\Xi
\]

本项目的结构发现路径为：

```text
partial observations
        ↓
short-history state observer
        ↓
explicit Pm dynamics
        ↓
minimal memory state q
        ↓
physics-constrained sparse ODE
        ↓
sparse energy / dissipation structure
        ↓
stability analysis and margin
```

核心研究贡献集中在：

1. 用短时历史恢复被低阶观测遗漏的 memory state；
2. 将 mechanical-power dynamics 显式从 latent dynamics 中分离；
3. 发现最小、稀疏的 post-fault reduced ODE；
4. 从发现的 ODE 中识别能量/耗散结构；
5. 基于显式模型计算稳定裕度。

---

## 17. 必做实验

只保留和主线直接相关的正式比较。

| Model | Pm dynamics | history observer | memory q | sparse energy |
|---|---|---|---|---|
| M2 | no | no | no | no |
| M3-Pm | yes | no | no | no |
| M4 | yes | yes | yes | no |
| M4-E | yes | yes | yes | yes |

每个模型评估：

- 0.2 / 0.5 / 1 / 2 / 5 / 8 s trajectory RMSE；
- pooled / mean / worst-10% CVaR；
- stable/unstable qualitative behavior；
- long-horizon boundedness；
- disturbance-group generalization；
- model sparsity；
- parameter consistency。

M4-E 额外评估：

- energy monotonicity / dissipation behavior；
- critical energy separation；
- stability-margin monotonicity；
- 与数值稳定边界的一致性。

History window、encoder 类型和 \(q\) 维度只做小规模必要消融，不为每一种组合建立独立模型框架。

---

## 18. 模型验收门

### Prediction

- mean RMSE 必须有明确改善；
- pooled RMSE 不恶化；
- worst-10% CVaR 不显著恶化；
- 2–8 s horizon 不产生新的非物理解发散。

### Physics

- \(M>0\)；
- damping 非负；
- \(T_m>0\)；
- \(P_m\)-ODE 具有稳定合理的慢动态；
- swing residual 保持在可接受范围；
- sparse energy 在 equilibrium 附近具有合理曲率。

### Structure

- dynamics 为显式方程；
- sparse terms 数量受控；
- \(q\) 维数从 1 开始；
- 新增第二个 \(q\) 必须有跨事件和稳定性收益；
- encoder 不参与 future rollout；
- 不使用 fault metadata 和 future leakage。

### Stability

进入最终稳定性分析前至少满足：

- discovered equilibrium 正确；
- Jacobian 模态合理；
- stable trajectories 的 energy behavior 一致；
- stability margin 与实际 stable/unstable 结果具有明确单调关系。

---

## 19. 推荐实施顺序

### Step 1：完成 Pilot 数据接入并建立 M2

只用于统一数据接口和基线指标。

### Step 2：完成 M3-Pm

直接利用真实 \(P_m(t)\) 拟合最低复杂度 \(P_m\)-ODE，并验证长时 horizon 收益。

### Step 3：完成 M4

在 M3-Pm 上加入 50 ms short-history observer，首先只使用一个 \(q_e\)。

### Step 4：稀疏化 M4

对 \(P_e\) closure、\(q_e\)-ODE 和必要的机械侧 residual 进行物理约束的 sparse discovery。

### Step 5：发现稀疏能量函数

建立小规模 energy library，识别 \(H(x_r)\)，验证 equilibrium 和耗散性质。

### Step 6：稳定性与裕度

完成：

- equilibrium；
- Jacobian/eigenvalue；
- energy margin；
- 数值 stability-boundary validation。

### Step 7：必要时增加模型阶次

只有 M4 在明确的 tail/cross-disturbance/stability 指标上不足时才尝试 \(q_{e,2}\)。

---

## 20. 论文级最终模型目标

期望得到类似：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= P_m-\frac{\partial U(\delta,q_e)}{\partial\delta}-D\omega,\\
T_m\dot P_m &= P_{ref}-P_m-K_\omega\omega+r_m,\\
T_q\dot q_e &= -q_e+g(\delta,\omega,P_m)
\end{aligned}
}
\]

其中：

- \(q_{e,0}\) 由短时 \(\delta,\omega\) 历史 observer 给出；
- future rollout 完全由显式 ODE 完成；
- \(U\)、\(g\) 和必要的 \(r_m\) 为物理约束下的稀疏发现结果；
- 在该 reduced ODE 上进一步发现 \(H(x_r)\) 并进行稳定性分析。

最终研究问题为：

> **能否从有限低阶观测的短时历史中恢复最小必要状态，并发现一个同时保留机械功率动态、能量结构与稳定性质的低阶 post-fault 电力系统模型？**
