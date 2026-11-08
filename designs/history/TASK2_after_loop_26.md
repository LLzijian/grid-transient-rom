# 任务二：记忆闭合、机械功率动力学与稀疏能量结构的低阶模型设计

## 1. 总体目标

基于任务一构建的私有 post-fault 数据集，建立一个从部分可观测高阶电力系统暂态中发现**低阶、显式、稀疏、具有记忆闭合并可进行稳定性分析的动力学模型**。

研究主线为：

\[
\boxed{
\text{short history}
\rightarrow
\text{minimal hidden state}
\rightarrow
\text{explicit reduced ODE}
\rightarrow
\text{sparse energy / stability structure}
}
\]

最终模型应满足：

1. \(P_m\) 采用显式低阶动力学描述，不长期使用 ZOH；
2. 使用短时 \(\delta,\omega\) 历史恢复单点观测中缺失的内部状态信息；
3. 黑盒仅用于状态初始化或状态观测，不直接负责长期 rollout；
4. future dynamics 必须由显式 reduced ODE 推进；
5. 新增 memory state 必须通过跨事件预测、尾部行为和物理合理性获得其复杂度；
6. 动力学结构优先采用可解释、低阶、稀疏、受物理约束的形式；
7. 在通过预测与结构验收的 reduced ODE 上进一步建立能量、耗散、Jacobian、局部模态、稳定边界和稳定裕度分析。

本任务不再验证 \((\delta,\omega)\) 是否构成严格 Markov state，而直接按照“**短时历史 + 最小记忆闭合 + 显式物理方程**”路线推进。

---

## 2. 当前开发基线与已完成工作

### 2.1 数据状态

任务一 Pilot 数据已完成并通过 QA：

- accepted trajectories：361；
- stable：305；
- physical unstable：56；
- numerical failure / solver divergence 等无效事件不进入正式数据集；
- post-fault 网络及设备配置保持一致；
- train / validation / frozen test 按预定义规则隔离。

后续 TASK2 的所有结构筛选继续只使用 development 数据；冻结 test 仅在模型结构最终确定后使用一次。

### 2.2 M2 clean baseline

M2 已建立并冻结。

其角色仅为：

- 统一的数据接口；
- 统一 rollout 及多时域评估；
- 提供不含显式 \(P_m\) 动态和 memory state 的低阶比较基准。

M2 长期预测明显不足，不再围绕 M2 继续增加补丁或搜索复杂二阶修正。

### 2.3 M3-Pm

M3-Pm 已通过当前开发阶段的辨识门和贡献门。

使用机械功率偏差：

\[
p=P_m-P_m^*
\]

当前通过验证的基本结构为：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-\mathcal E(\delta)-D\omega,\\
\dot p &= -A_m p-B_m\omega.
\end{aligned}
}
\]

其中：

- \(\mathcal E(\delta)\) 为相对 post-fault equilibrium 归零的电气耦合；
- \(A_m\) 为正的恢复速率；
- \(B_m\) 描述速度偏差对机械功率动态的反馈；
- M3 参数在进入 M4 后冻结，除非新的结构诊断明确证明 M3 本身存在错误。

当前 development evidence：

- 8 s normalized trajectory pooled RMSE：27.49 → 0.464；
- 70 个可预测 validation 事件中未观察到稳定性误判；
- 尚未使用 frozen test，因此当前结论属于开发验证，不定义为最终泛化结论。

### 2.4 M4 当前科学筛选

M4 已尝试多种具体 memory 方案，包括：

- additive memory；
- dissipation-constrained memory；
- instantaneous-response separation；
- 其他低阶恢复结构。

这些具体实现或产生非物理行为，或收益不足，均已按预注册规则回退。

曾得到约 0.972 的 initial-memory estimation correlation，说明：

> “从历史中估计某个初始隐藏量”与“找到正确的后续 hidden-state dynamics”是两个不同问题。

因此后续不以 encoder 拟合好坏代替 memory closure 的动力学验证。

---

## 3. 当前 M4 候选模型：一维电气耦合强度恢复

当前 Loop 0026 检验最小的一维 memory model。

10 台机组共享一个全局标量：

\[
q\in\mathbb R.
\]

总状态维数为：

\[
10\;\delta+10\;\omega+10\;P_m+1\;q=31.
\]

当前方程：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-e^q\mathcal E(\delta)-D\omega,\\
\dot p &= -A_m p-B_m\omega,\\
\dot q &= -a_q q,\qquad a_q>0.
\end{aligned}
}
\]

物理解释暂定为：

- \(q<0\)：故障清除后有效电气耦合暂时减弱；
- \(q>0\)：有效电气耦合暂时增强；
- \(q\to0\)：内部电压/控制状态恢复后回到 M3 电气耦合；
- 当前 \(q\) 仅视为 effective electrical/control memory state，不声明其等于某个单一真实 ANDES state。

该模型实际检验的是：

> **单一时间尺度、全系统共享、无持续驱动的同步耦合强度恢复，是否足以解释 M3-Pm 剩余误差。**

它不是一般意义下最终的 electrical memory closure。

---

## 4. 当前 history observer

### 4.1 观察窗口

故障清除后先观察：

\[
T_{obs}=50\text{ ms}
\]

共 6 个采样点。

### 4.2 输入特征

当前使用 40 个结构化历史特征：

- 当前 COI angle：10；
- 当前 speed deviation：10；
- 窗口 mean speed：10；
- 窗口 speed trend：10。

标准化后采用线性 observer：

\[
q_0=
\operatorname{clip}
\left(
\beta^Tf_{history},
-3,
\log2
\right).
\]

### 4.3 功能边界

Observer 只执行一次初始化。

不允许：

- 在 rollout 中持续读取新观测；
- 直接输出 future trajectory；
- 使用 fault line / duration / location / type 等 metadata；
- 使用 future \(P_e\)、future \(P_m\) 或 future trajectory；
- 通过大规模 neural network 掩盖 ODE 结构不足。

当前只学习：

- 40 个 observer weights；
- 1 个 recovery rate \(a_q\)。

M3 参数全部冻结。

---

## 5. Loop 0026 的执行与验收

Loop 0026 不重新设计模型族，只完成当前候选的公平优化与科学筛选。

上一轮优化出现训练目标显著恶化，因此当前采用带线搜索的 L-BFGS，并已通过梯度检查。

本轮至少记录：

1. initial / final train objective；
2. validation pooled RMSE；
3. validation mean RMSE；
4. worst-10% CVaR；
5. stable / unstable 分层结果；
6. long-horizon boundedness；
7. \(q_0\) 分布及 clipping ratio；
8. \(a_q\) 是否撞到边界；
9. 最差事件及其物理轨迹；
10. 与冻结 M3-Pm 的逐事件差异。

基本优化门：

\[
\mathcal L_{train,final}
\le
\mathcal L_{train,initial}.
\]

若连训练目标都不能降低，则该具体参数化/优化组合直接失败，不进入更高层预测解释。

Loop 0026 结束后必须形成 accepted 或 failed evidence，并建立清晰 Git checkpoint。

---

## 6. M4 的核心判断原则

M4 的目标不是“必须证明 memory 有用”，而是：

> 在已经成功显式建模 \(P_m\) 后，剩余 high-order electrical/control dynamics 是否能够由一个最小、可解释的 memory state 稳定闭合。

因此：

- 单个 memory 方案失败，不等于 memory concept 失败；
- encoder 可以很好地估计 \(q_0\)，也不等于 \(q\)-ODE 正确；
- average RMSE 改善但严重损害 tail，不算通过；
- 只有新增 state 同时改善 prediction、tail、physics 或 stability 中的核心指标，才保留其复杂度；
- 不因为理论上高阶系统存在 hidden states，就强制最终模型必须含 latent state。

如果经过合理的一维 memory closure 筛选后 M3-Pm 仍显著更稳健，则允许 M3-Pm 成为最终主要动力学模型。

---

## 7. Loop 0026 后的诊断顺序

无论 0026 pass 或 fail，都不立即增加第二个 \(q\) 或切换 GRU。

### 7.1 Oracle initial-state test

固定候选 \(q\)-ODE 与 M3-Pm，对每个 development event 独立求：

\[
q_{0,i}^{oracle}
=
\arg\min_{q_0}
\mathcal L_i(q_0).
\]

该实验只用于诊断，不作为正式预测模型。

目的：区分

- observer capacity 不足；
- memory state dynamics / coupling form 不足。

#### 判据

若 oracle \(q_0\) 仍无法显著优于 M3：

> 当前一维 coupling-strength model family 的表达能力不足，不应继续升级 observer。

若 oracle \(q_0\) 显著优于 M3，而 history observer 明显追不上：

> 当前状态方程可能有价值，后续才允许优化 observer。

### 7.2 Time-resolved oracle memory inversion

直接利用 full-order trajectory 反演每个时刻所需的最优 coupling modifier。

令 M3 剩余电气 residual 为：

\[
r_e(t).
\]

对当前候选结构求最优标量 \(s(t)\)：

\[
r_e(t)\approx s(t)\,\mathcal E(\delta(t)).
\]

并转换为：

\[
q_{oracle}(t)=\log(1+s(t))
\]

（仅在数值有效、定义合理的区域计算）。

检查：

- 是否近似单调恢复；
- 是否近似单指数；
- 是否存在明显 overshoot；
- 是否变号；
- 是否存在多个时间尺度；
- 不同 disturbance groups 是否需要完全不同的规律；
- 单一全局 \(q\) 是否足以解释 generator-wise residual。

该诊断决定后续 \(q\)-ODE 的结构，而不是盲目增加网络容量。

---

## 8. 电气耦合结构的下一层候选

当前 centered-strength form：

\[
e^q\left[F(\delta)-F(0)\right]
\]

只改变相对 equilibrium 的 restoring force，因此在 \(\delta=0\) 时，\(q\) 无法产生瞬时 electrical power offset。

如果 oracle 诊断显示 hidden electrical state 主要表现为“内部电压/传输能力暂时变化”，下一层优先检验未中心化强度恢复：

\[
\boxed{
P_e^{dev}(\delta,q)
=
e^q F(\delta)-F(0)
}
\]

其中 \(F(0)\) 为 post-fault equilibrium 下的基准电气功率。

该形式满足：

- \(q=0\) 时严格回到 M3；
- \(q\to0\) 时最终 equilibrium 与 M3 一致；
- \(q\neq0\) 时即使 \(\delta=0\)，仍允许 transient electrical power mismatch；
- 仍只增加一个标量 \(q\)，不增加 observer 复杂度。

该模型应作为“内部电压/传输能力恢复”的最小物理候选，而不是直接进入更高维 latent state。

---

## 9. 一维 driven memory

如果 time-resolved oracle 明确显示 \(q\) 不是纯自由衰减，而是持续受到当前状态驱动，则进入一维 driven memory。

统一形式：

\[
\boxed{
T_q\dot q
=
-q+g(\delta,\omega,p,q)
}
\]

第一阶段保持 \(g\) 为小规模稀疏候选库。

优先候选：

\[
q,
\qquad
\omega,
\qquad
p,
\qquad
\sin(B^T\delta),
\qquad
\cos(B^T\delta)-\cos(B^T\delta^*).
\]

只允许少量低阶 cross terms。

原则：

1. 不开放任意高阶 polynomial library；
2. 不先用 MLP 生成 \(\dot q\)；
3. 使用 sparsity / threshold / cross-event consistency 保留最小 closure；
4. 只有新增 driven terms 同时带来明确 prediction / tail / physics 收益才保留。

目标是发现：

\[
\dot q=f_q(\delta,\omega,p,q)
\]

的最小显式结构，而不是把 \(q\) 变成不可解释的 latent neural dynamics。

---

## 10. Memory observer 的升级条件

Observer 的复杂化必须晚于 memory dynamics 的结构诊断。

优先顺序：

1. 当前 compact 50 ms history features；
2. 低维 delay / trend features；
3. small GRU；
4. small TCN。

只有在 oracle \(q_0\) 明显有效而当前 observer 无法恢复时，才允许 small GRU/TCN。

若使用神经 observer：

- hidden dimension 优先 8 或 16；
- output 仍为 \(q_0\)；
- observer 只运行一次；
- future rollout 不再调用网络；
- train/validation 分组隔离；
- 必须报告 observer-only accuracy 与 end-to-end rollout accuracy，防止“初态估计好但动力学不成立”的混淆。

---

## 11. Memory state 维数扩展规则

默认：

\[
\dim(q)=1.
\]

只有以下条件同时满足时，才允许增加 \(q_2\)：

1. 一维 driven memory 已完成充分结构搜索；
2. oracle diagnostics 显示残差中仍存在第二个独立、可重复的时间尺度/空间模式；
3. 单一 \(q\) 在多个 disturbance groups 上表现出一致的结构性不足；
4. 二维 memory 在 validation 上改善 pooled / mean，同时不牺牲 worst-tail；
5. 新维度能获得可解释的 mode / time-scale / full-state association。

不得仅因为训练 loss 还能下降就增加 state dimension。

---

## 12. M3-Pm 的能量结构基线

能量/稳定性工作不需要等待 M4 最终成功才开始。

M3-Pm 已具有可直接推导的储能—耗散结构，应立即建立为后续所有 M4 方案的结构基线。

定义：

\[
p=P_m-P_m^*.
\]

M3：

\[
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-\nabla U(\delta)-D\omega,\\
\dot p &= -A_mp-B_m\omega.
\end{aligned}
\]

若 \(B_m\succ0\)，定义：

\[
\boxed{
H_3
=
\frac12\omega^TM\omega
+U(\delta)
+\frac12p^TB_m^{-1}p
}
\]

则：

\[
\boxed{
\dot H_3
=
-\omega^TD\omega
-p^TB_m^{-1}A_mp
\le0.
}
\]

### 12.1 网络 potential

对于正弦耦合，\(U(\delta)\) 应显式由已冻结 M3 电气项积分得到，使：

\[
\nabla U(\delta)=\mathcal E(\delta).
\]

候选形式可按 edge sum 构造：

\[
U(\delta)
=
\sum_eK_e
\left[
-\cos\eta_e
+\cos\eta_e^*
-\sin\eta_e^*(\eta_e-\eta_e^*)
\right].
\]

### 12.2 M3 energy validation

在 development trajectories 上验证：

- equilibrium 附近 \(H_3\) 是否局部正定；
- 数值 \(\dot H_3\) 与理论耗散项是否一致；
- stable trajectories 上是否总体耗散；
- solver / interpolation error 是否造成明显假正增长；
- M3 的能量值与 disturbance severity、稳定/失稳结果的关系。

该阶段不定义最终 critical energy，不提前宣称最终 stability margin。

---

## 13. M4 的稀疏能量结构

M4 通过预测与物理验收后，在 \(H_3\) 基础上扩展，而不是从零训练一个任意 neural Lyapunov function。

目标形式：

\[
\boxed{
H_4
=
H_3
+H_q(q)
+H_c(\delta,\omega,p,q)
}
\]

优先候选项：

- \(q^2\)；
- \(q\sin(B^T\delta)\)；
- \(q[1-\cos(B^T\delta)]\)；
- \(qp\)；
- 必要的低阶 physically plausible coupling terms。

保持小型 library：

\[
H_4(x)=\Theta_H(x)\xi_H.
\]

要求：

1. equilibrium 附近有局部 minimum；
2. M4 dynamics 与 \(\nabla H_4\) 具有一致的 storage / dissipation interpretation；
3. stable trajectories 不出现大范围无解释能量增长；
4. 若 \(q\) 可释放暂存能量，应通过 \(H_q/H_c\) 显式解释，而不是简单禁止 \(H_3\) 增长；
5. sparse energy structure 不以显著牺牲 rollout accuracy 为代价。

完整 port-Hamiltonian：

\[
\dot x=[J(x)-R(x)]\nabla H(x)
\]

以及：

\[
J=-J^T,
\qquad
R=R^T\succeq0
\]

仅在低复杂度 \(H_4\) 和显式 M4 ODE 已得到数据支持后再尝试，不作为 M4 的前置开发框架。

---

## 14. 稀疏结构发现

最终论文模型不能以：

\[
\dot x=NN(x)
\]

作为主要动力学。

结构发现集中于：

1. \(P_m\)-ODE 的必要 residual；
2. electrical coupling 中 \(q\) 的作用形式；
3. \(q\)-ODE 的 driving terms；
4. sparse energy/storage terms。

### 14.1 机械侧

M3-Pm 已通过后默认冻结：

\[
\dot p=-A_mp-B_m\omega.
\]

只有在 M4 残差诊断明确显示机械侧存在系统性未建模项时，才允许增加：

\[
\Theta_m(p,\omega)\xi_m.
\]

### 14.2 电气侧

优先保持正弦/网络结构，不采用大规模多项式拟合 \(P_e\)。

候选包括：

\[
\sin(B^T\delta),
\quad
\cos(B^T\delta),
\quad
q\sin(B^T\delta),
\quad
q\cos(B^T\delta).
\]

### 14.3 Memory side

优先：

\[
q,
\quad
\omega,
\quad
p,
\quad
\sin(B^T\delta),
\quad
\cos(B^T\delta)-\cos(B^T\delta^*).
\]

使用 constrained / sparse regression，只保留跨事件稳定且物理符号合理的少量项。

---

## 15. 损失函数与训练原则

### 15.1 M3-Pm

M3 已完成，后续不重新联合优化。

### 15.2 M4 screening

每个 M4 candidate 使用尽量少的目标：

\[
\mathcal L
=
\lambda_{traj}\mathcal L_{traj}
+
\lambda_{tail}\mathcal L_{tail}
+
\lambda_{reg}\mathcal L_{reg}.
\]

必要时增加明确的 physics residual，而不是一次堆叠多个软约束。

### 15.3 Event-balanced / tail-aware

正式模型必须：

- event-balanced weighting；
- 监控 worst-k；
- 监控 worst-10% CVaR；
- stable / unstable 分层；
- disturbance-group 分层；
- 长时域 boundedness。

平均性能改善不能通过牺牲少数 severe events 获得。

### 15.4 Optimization discipline

每个候选必须先验证：

- finite forward；
- finite gradient；
- directional derivative；
- line search / trust-region 行为；
- train objective 确实可下降。

优化失败与模型族失败必须分开记录。

---

## 16. 模型族与正式开发节点

只维护以下正式层级：

| Model | State | Status / Role |
|---|---|---|
| M2 | \([\delta,\omega]\) | 已完成并冻结；baseline |
| M3-Pm | \([\delta,\omega,P_m]\) | 已通过 development validation；核心物理基线 |
| M4-1 | \([\delta,\omega,P_m,q]\) | 当前科学筛选；一维 memory closure |
| M4-S | M4-1 + sparse driven closure | 一维 memory 结构发现候选 |
| M4-E | accepted M4 + sparse energy/storage | 能量与稳定性目标模型 |

不为每一个实验小变体建立独立长期框架。

每个 loop 只回答一个明确科学问题，并在 pass / fail 后关闭。

---

## 17. M4 决策树

### 情况 A：Loop 0026 通过

保留当前 recovering-strength model 作为 M4 candidate，但继续完成：

1. \(q_0\) saturation 检查；
2. \(a_q\) 边界检查；
3. oracle \(q_0\) gap；
4. time-resolved \(q_{oracle}(t)\)；
5. stable / unstable / severe group 分层；
6. centered strength 的局限诊断；
7. 与 M3 energy structure 的兼容性。

若这些结果支持单指数 recovering-strength，则进入 sparse energy extension。

### 情况 B：Loop 0026 未通过，但 oracle \(q_0\) 有明显收益

说明 state family 可能成立，observer 不足。

按顺序：

1. 改善 compact history features；
2. small GRU/TCN；
3. 保持同一显式 \(q\)-ODE；
4. 重新做 end-to-end gate。

### 情况 C：Loop 0026 未通过且 oracle \(q_0\) 也无明显收益

拒绝当前 centered recovering-strength family。

优先测试：

\[
e^qF(\delta)-F(0).
\]

若 oracle trajectory 显示持续驱动，再进入：

\[
T_q\dot q=-q+g(\delta,\omega,p,q).
\]

### 情况 D：一维 driven memory 仍不足

先通过 oracle residual / PCA / full-state association 证明存在第二个稳定独立 mode，再允许 \(q_2\)。

不直接从失败的一维自由衰减跳到高维 neural latent model。

---

## 18. Memory state 的物理解释

对通过验证的 \(q\)，使用任务一保存的 full-order states 做后验解释。

优先检查：

- transient EMF；
- internal voltage；
- AVR states；
- governor/control states；
- \(P_e\)；
- terminal voltage；
- generator-wise electrical residual；
- full-state PCA / dominant subspace。

方法：

- Pearson / rank correlation；
- linear projection；
- simple regression；
- PCA / subspace alignment；
- time-scale comparison。

若 \(q\) 与某个物理组合高度一致，可给出更具体解释；否则继续定义为：

> unresolved electrical/control memory state。

不得为了叙事强行把 \(q\) 命名成某一个真实高阶状态。

---

## 19. 稳定性分析

稳定性分析只针对最终显式 reduced ODE：

\[
\dot x_r=F_r(x_r).
\]

History observer 不参与 future dynamics，因此不进入 autonomous stability equation。

### 19.1 Equilibrium

求：

\[
F_r(x^*)=0
\]

并验证与统一 post-fault equilibrium 一致。

### 19.2 Local stability

计算：

\[
A=
\left.
\frac{\partial F_r}{\partial x}
\right|_{x^*}.
\]

至少输出：

- dominant eigenvalue；
- damping ratio；
- modal frequency；
- state participation / sensitivity；
- 与 full-order 主导模态的趋势一致性。

### 19.3 Nonlinear energy stability

首先以：

\[
H_3
\]

作为 M3 基线。

若 M4 accepted，则使用：

\[
H_4=H_3+H_q+H_c.
\]

检查：

- equilibrium minimum；
- stable trajectories 的 storage/dissipation behavior；
- unstable trajectories 是否跨越 candidate energy boundary；
- 不同 disturbance groups 的一致性。

### 19.4 Numerical ROA / boundary

数值 basin / ROA 扫描只用于验证显式模型和 energy boundary，不提前搭建大型通用 ROA framework。

---

## 20. 稳定裕度

最终优先使用经典能量/谱稳定概念。

### 20.1 Energy margin

若可以可靠定义 critical energy：

\[
\boxed{
\Delta H
=
H_{crit}-H(x_r)
}
\]

### 20.2 Normalized energy margin

\[
\eta_H
=
\frac{H_{crit}-H(x_r)}{H_{crit}}.
\]

### 20.3 Local spectral margin

\[
\eta_\lambda
=
-\max_i\Re(\lambda_i).
\]

### 20.4 Numerical boundary distance

必要时在 reduced-state 空间计算：

\[
d(x_r,\partial\mathcal A).
\]

该指标用于辅助验证，不替代 energy margin 主线。

---

## 21. 稳定域的研究对象

若最终 accepted model 为 M3：

\[
x_r=[\delta,\omega,P_m].
\]

若最终 accepted model 为 M4：

\[
x_r=[\delta,\omega,P_m,q].
\]

因此严格稳定域是 reduced-state full space 中的：

\[
\mathcal A\subset\mathbb R^{\dim(x_r)}.
\]

二维 \(\delta-\omega\) 图只用于：

- 固定其余 reduced states 的切片；
- 低维投影；
- 工程可视化；
- 与传统暂态稳定图形对照。

不强制把高维稳定性压缩成天然唯一的二维硬边界。

---

## 22. 必做实验

### 22.1 已完成基线

- M2 clean baseline；
- M3-Pm identification；
- M3-Pm long-horizon contribution evaluation。

### 22.2 当前必须完成

- Loop 0026 final optimization / validation；
- oracle \(q_0\)；
- time-resolved \(q_{oracle}(t)\)；
- M3 energy derivation and numerical validation。

### 22.3 后续按证据触发

- uncentered electrical-strength candidate；
- one-dimensional driven memory；
- small neural observer；
- second memory state；
- sparse energy extension；
- stability margin validation。

### 22.4 统一评估指标

所有正式模型输出：

- 0.2 / 0.5 / 1 / 2 / 5 / 8 s trajectory RMSE；
- pooled RMSE；
- mean RMSE；
- worst-10% CVaR；
- stable / unstable qualitative behavior；
- long-horizon boundedness；
- disturbance-group generalization；
- parameter signs / scales；
- model sparsity / state dimension。

最终 M4-E 额外输出：

- energy/storage consistency；
- \(\dot H\) behavior；
- critical-energy separation；
- energy-margin monotonicity；
- numerical stability-boundary agreement。

---

## 23. 模型验收门

### 23.1 Prediction

新增结构至少满足：

- pooled RMSE 有明确收益或不恶化；
- mean RMSE 有明确收益；
- worst-10% CVaR 不出现不可接受恶化；
- 2–8 s 不新增非物理解发散；
- severe events 不因优化平均指标而系统性牺牲。

### 23.2 Physics

- \(M>0\)；
- damping 非负；
- \(A_m>0\)；
- M3 \(P_m\)-ODE 保持稳定慢动态；
- \(a_q>0\)（自由恢复候选）；
- driven memory 的 linear recovery component 稳定；
- electrical coupling 恢复到统一 M3 post-fault equilibrium；
- 参数不依赖 clipping / boundary saturation 才能工作。

### 23.3 Structure

- future dynamics 为显式 ODE；
- observer 只初始化；
- sparse terms 数量受控；
- \(q\) 从 1 维开始；
- 第二个 \(q\) 必须有独立物理/数据证据；
- 不使用 fault metadata；
- 不使用 future leakage；
- 不因一次优化失败直接判定整个 model family 失败。

### 23.4 Energy / stability

进入最终稳定裕度前至少满足：

- equilibrium 正确；
- Jacobian 模态合理；
- M3/M4 storage function 在平衡点附近曲率合理；
- stable trajectories 的 energy behavior 可解释；
- candidate margin 与 stable/unstable 或 distance-to-boundary 具有明确单调关系。

---

## 24. 推荐实施顺序

### Step 1：关闭 Loop 0026

完成当前一维 recovering-strength model 的线搜索优化、验证和 pass/fail evidence。

### Step 2：做两个低成本 oracle 诊断

完成：

\[
q_0^{oracle}
\]

与：

\[
q_{oracle}(t).
\]

明确瓶颈属于 observer、free-decay ODE、centered strength form，还是全局单标量假设。

### Step 3：建立 M3 energy baseline

推导并数值验证：

\[
H_3
=
\frac12\omega^TM\omega
+U(\delta)
+\frac12p^TB_m^{-1}p.
\]

将其作为以后所有 M4 energy extension 的基准。

### Step 4：按诊断选择一维 M4 结构

优先级：

1. 当前 centered recovering strength；
2. uncentered/effective-voltage strength；
3. one-dimensional driven memory。

每次只改变一个结构假设。

### Step 5：必要时升级 observer

只有 oracle state 明确有用而当前 history observer 不足时，再使用 small GRU / TCN。

### Step 6：必要时增加第二 memory mode

只有一维 driven memory 已充分失败且 residual 中存在第二个稳定 mode 时，才增加 \(q_2\)。

### Step 7：稀疏化 accepted M4

对：

- electrical \(q\)-coupling；
- driven \(q\)-ODE；
- 必要的 mechanical residual

做 physics-constrained sparse discovery。

### Step 8：发现 M4 sparse energy/storage

在 \(H_3\) 上增加最小 \(H_q/H_c\)，验证 energy / dissipation consistency。

### Step 9：稳定性与裕度

完成：

- equilibrium；
- Jacobian / eigenvalues；
- local spectral margin；
- energy margin；
- numerical boundary / ROA validation。

### Step 10：冻结模型并使用 test

只有结构、参数、observer、energy/stability definition 全部冻结后，才打开 frozen test 一次。

---

## 25. 论文级目标模型

理想情况下，最终得到：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-\frac{\partial U(\delta,q)}{\partial\delta}-D\omega,\\
\dot p &= -A_mp-B_m\omega+r_m,\\
T_q\dot q &= -q+g(\delta,\omega,p,q),
\end{aligned}
}
\]

其中：

- \(q_0\) 由 50–200 ms short-history observer 初始化；
- future rollout 完全由显式 ODE 完成；
- \(r_m\) 默认等于 0，仅在 M3 residual 有强证据时保留；
- \(U\)、\(g\) 与必要 coupling terms 通过物理约束 sparse discovery 获得；
- M3 的机械侧和网络能量结构尽量保留；
- M4 新增的 storage/coupling 通过 \(H_q/H_c\) 显式表示；
- 稳定裕度建立在最终显式 reduced ODE 及其 energy structure 上。

若一维/低维 memory 经合理筛选仍不能稳定超越 M3-Pm，则允许最终主要模型退化为：

\[
\boxed{
[\delta,\omega,P_m]
}
\]

并将“额外 latent memory 对当前 post-fault reachable manifold 的增益有限”作为数据支持的研究结论，而不是强制保留无收益的复杂状态。

---

## 26. 最终研究问题

本任务最终回答：

> **在机械功率慢动态已显式分离的前提下，能否利用短时低阶观测恢复最小必要的电气/控制记忆状态，并进一步发现一个显式、稀疏、具有储能—耗散结构、能够进行稳定裕度分析的 post-fault reduced-order model？**

同时回答两个次级问题：

1. 一维 effective memory state 是否足以解释 M3-Pm 剩余的跨扰动误差；
2. 若 memory state 被数据支持，它是否能够与网络电气能量、机械侧储能及稳定边界形成统一的结构化解释。
