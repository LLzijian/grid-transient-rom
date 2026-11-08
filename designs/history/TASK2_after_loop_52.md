# TASK2_after_loop_52：物理拓扑约束的联合潜状态—稀疏动力学发现方案

## 1. 研究目标

本任务面向 IEEE-39 post-fault 暂态，目标是在已验证的机械侧物理骨架上，进一步发现能够解释高阶电气/控制记忆的**低维潜状态、拓扑空间作用结构及其显式稀疏 ODE**。

研究主线调整为：

\[
\boxed{
\text{short-history observations}
\rightarrow
\text{latent electrical state }q
\rightarrow
\text{topological electrical action}
\rightarrow
\text{sparse ODE discovery}
\rightarrow
\text{autonomous rollout}
\rightarrow
\text{energy/stability analysis}
}
\]

核心思想不是先人工定义某个 memory state，再单独拟合其 ODE；而是借鉴 SINDy-SHRED 思路，将：

1. 潜状态 \(q\) 的生成；
2. \(q\) 的显式稀疏动力学；
3. \(q\) 对电气力的拓扑空间作用；
4. 长时 autonomous rollout；

放在同一个优化框架中联合发现。

最终希望回答：

> 能否从有限历史观测中自动发现少量具有物理空间结构、可由稀疏 ODE 描述、并能改善长期暂态预测与稳定性几何的电气记忆状态？

---

## 2. 已完成基础与当前结论

以下结果视为后续固定基础，不重复重新开发。

### 2.1 数据

- Pilot 及后续扩展数据链已建立；
- post-fault 系统一致；
- stable / physical unstable 分开；
- numerical failures 不进入正式数据；
- train / validation / frozen test 采用 disturbance-group 隔离；
- 已保存 \(\delta,\omega,P_m,P_e\) 及 full-order dynamic/algebraic states。

### 2.2 M2

M2 已冻结，仅作为最基础比较模型。

### 2.3 M3-Pm

保留：

\[
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-\mathcal E_0(\delta)-D\omega,\\
\dot p &= -A_mp-B_m\omega,
\end{aligned}
\]

其中：

\[
p=P_m-P_m^*.
\]

M3-Pm 已证明机械功率慢动态具有明确预测贡献，并在冻结测试上显著改善长期轨迹预测。

但 M3 不作为最终研究终点，而作为：

\[
\boxed{\text{physical backbone / scaffold}}
\]

后续所有电气记忆结构均建立在该骨架上。

### 2.4 M4 既有筛选结论

已尝试：

- 全系统共享 scalar memory；
- additive memory；
- passive / dissipative memory；
- centered / uncentered coupling-strength memory；
- driven memory；
- local-stability constrained memory；
- small GRU observer；
- rank-1 spatial residual modes；
- load-power-based memory coordinate。

这些方案均未同时满足 prediction、tail、stability 与 cross-disturbance generalization。

现有结果支持以下判断：

1. M3 仍存在显著未建模 electrical dynamics；
2. full residual force 具有很强的 oracle predictive capacity；
3. 单一 rank-1 / global scalar memory 不足；
4. pointwise force fit 不能代替 autonomous dynamics；
5. residual 中存在真实 network/load port contribution；
6. \(P_{load}-P_{load}^*\) 更适合作为 algebraic output，而不是独立 memory state；
7. hidden GENROU / AVR / internal-voltage states 与 residual 存在显著关联；
8. “先找 \(q\)，再拟合 ODE”容易得到无法闭合的坐标。

---

## 3. 新的核心建模原则

### 3.1 不再预定义 \(q\) 的物理形式

不再预先规定：

\[
q(t)=q_0e^{-at}
\]

或：

\[
P_e=e^qP_{e,0}.
\]

这些只保留为历史基线，不再作为主路线。

新的目标是联合发现：

\[
\boxed{q,\quad C,\quad \Xi_q}
\]

其中：

- \(q\)：低维 electrical/control memory state；
- \(C\)：memory state 的拓扑空间作用模式；
- \(\Xi_q\)：SINDy 稀疏动力学系数。

### 3.2 保留已知物理状态

不采用完全黑盒 latent model。

Reduced state 明确写为：

\[
\boxed{x_r=[\delta,\omega,p,q]}
\]

其中：

- \(\delta,\omega,p\) 保持物理含义；
- 仅 \(q\) 为待发现 latent electrical state。

### 3.3 黑盒只用于坐标发现与初始化

Encoder 可使用 GRU / compact temporal encoder，但其角色仅为：

\[
\boxed{\mathcal H_t\rightarrow q_t}
\]

它不直接生成 future trajectory。

预测阶段仅使用一次：

\[
q_0=E_\theta(\mathcal H_0).
\]

随后 future dynamics 完全由显式 ODE 推进。

---

## 4. 总体模型结构

### 4.1 机械与摆动骨架

保持：

\[
\dot\delta=\omega,
\]

\[
\dot p=-A_mp-B_m\omega.
\]

摆动方程改写为：

\[
\boxed{
M\dot\omega
=
p
-\mathcal E_0(\delta)
-D\omega
-r_{alg}(\delta,q)
-r_{mem}(q)
}
\]

其中：

- \(\mathcal E_0(\delta)\)：M3 已有 post-fault baseline electrical coupling；
- \(r_{alg}\)：由当前状态即时决定的 network/load algebraic correction；
- \(r_{mem}\)：真正由 latent electrical dynamics 引起的 memory correction。

### 4.2 拓扑空间 memory

不再使用全系统共享倍率。

定义完整网络 incidence / topology operator：

\[
B_{net}.
\]

Generator terminal selector：

\[
S_G.
\]

Memory force 优先写为：

\[
\boxed{
r_{mem}=S_GB_{net}Cq
}
\]

其中：

\[
q\in\mathbb R^r.
\]

\(C\) 决定每个 latent mode 在网络 edge / region 上的空间作用方向。

### 4.3 Load / network algebraic port

负荷功率不再作为独立 dynamic memory state。

优先表达：

\[
P_L=\Psi_L(\delta,q)
\]

或：

\[
V=\Psi_V(\delta,q),
\qquad
P_L=G_LV^2.
\]

第一阶段不要求精确重建完整 DAE，只要求该代数输出能够解释此前 residual 中被确认的 load-port contribution。

---

## 5. History Encoder 与潜状态生成

### 5.1 输入

训练阶段，对每个时间点 \(t\) 构造历史窗口：

\[
\mathcal H_t=
\{y(\tau)\}_{\tau=t-T_{obs}}^t.
\]

基础观测：

\[
y=[\delta,\omega].
\]

后续可比较：

\[
[\delta,\omega,P_e],
\]

\[
[\delta,\omega,V,I].
\]

是否增加观测变量由 observability experiment 决定，不直接加入 fault metadata。

### 5.2 Encoder

第一版：

\[
\boxed{q_t=E_\theta(\mathcal H_t)}
\]

优先使用 small GRU：

- hidden dimension：8 或 16；
- output dimension：\(r=1\) 起步；
- 不输入 absolute time；
- 不输入 fault bus / location / duration / label；
- 不输入 future information。

### 5.3 Latent normalization

防止 latent collapse 或任意尺度漂移：

\[
\operatorname{mean}(q_j)\approx0,
\]

\[
\operatorname{Var}(q_j)\approx1.
\]

若 \(r>1\)，增加弱 decorrelation / orthogonality constraint。

---

## 6. SINDy latent dynamics

### 6.1 核心形式

对 latent memory state：

\[
\boxed{
\dot q
=
\Theta_q(\delta,\omega,p,q)\Xi_q
}
\]

### 6.2 候选函数库

函数库保持物理驱动、小规模。

优先包括：

\[
1,
\quad
q,
\quad
\omega,
\quad
p,
\]

\[
\sin(B_{net}^T\delta),
\quad
\cos(B_{net}^T\delta)-\cos(B_{net}^T\delta^*),
\]

少量低阶交叉项：

\[
q\omega,
\quad
qp,
\quad
q\sin(B_{net}^T\delta).
\]

禁止第一轮直接开放大规模高阶 polynomial library。

### 6.3 连续时间积分

不直接通过 noisy numerical differentiation 拟合 \(\dot q\)。

训练中采用可微 ODE integration：

\[
q_{t+\Delta t}^{SINDy}
=
\operatorname{Integrate}
\left(
q_t,
\Theta_q\Xi_q,
\Delta t
\right).
\]

第一版可使用 RK4 或稳定小步积分，不固定使用 Euler。

### 6.4 Dynamics consistency

Encoder 在相邻窗口产生：

\[
q_t^{enc},
\qquad
q_{t+\Delta t}^{enc}.
\]

SINDy 从：

\[
q_t^{enc}
\]

积分得到：

\[
\hat q_{t+\Delta t}^{dyn}.
\]

动力学一致性损失：

\[
\boxed{
\mathcal L_{dyn}
=
\left\|
q_{t+\Delta t}^{enc}
-
\hat q_{t+\Delta t}^{dyn}
\right\|^2
}
\]

---

## 7. 电气结构约束

仅要求 latent q 具有简单 ODE 不足以保证其有物理作用，因此必须同时约束其 electrical relevance。

### 7.1 Residual force consistency

定义 M3 未解释的 electrical residual：

\[
r_e^{true}
=
M\dot\omega^{FOM}
-p
+\mathcal E_0(\delta)
+D\omega.
\]

模型给出：

\[
\hat r_e
=
r_{alg}(\delta,q)
+S_GB_{net}Cq.
\]

定义：

\[
\boxed{
\mathcal L_{force}
=
\|r_e^{true}-\hat r_e\|^2
}
\]

该损失只作为结构辅助，不作为最终 acceptance criterion。

### 7.2 Electrical output reconstruction

为避免 \(q\) 只拟合 force residual，可增加浅层物理解码器：

\[
\boxed{
\hat z_e
=D_\phi(\delta,\omega,p,q)
}
\]

第一阶段只重构与研究问题直接相关的量：

\[
z_e=[P_e,V,P_L]
\]

以及必要的少数已知高相关 high-order electrical states。

不要求重构全部 220 维 full state。

定义：

\[
\mathcal L_{elec}
=
\|z_e-\hat z_e\|^2.
\]

---

## 8. Autonomous rollout loss

这是最高优先级指标。

从某个 history endpoint \(t_0\) 初始化：

\[
q_0=E_\theta(\mathcal H_{t_0}).
\]

之后不再调用 encoder。

使用完整 explicit reduced ODE：

\[
[\delta,\omega,p,q]_{t_0}
\rightarrow
[\hat\delta,\hat\omega,\hat p,\hat q]_{t>t_0}.
\]

轨迹损失：

\[
\boxed{
\mathcal L_{roll}
=
\sum_t
w_t
\left(
\|\hat\delta-\delta\|^2
+
\|\hat\omega-\omega\|^2
\right)
}
\]

必要时加入 \(P_m\) 辅助项。

所有新结构必须以 autonomous rollout 为最终依据，禁止用 pointwise force R² 代替模型验收。

---

## 9. 总损失

第一版统一目标：

\[
\boxed{
\mathcal L
=
\lambda_{roll}\mathcal L_{roll}
+
\lambda_{dyn}\mathcal L_{dyn}
+
\lambda_{force}\mathcal L_{force}
+
\lambda_{elec}\mathcal L_{elec}
+
\lambda_{sp}\mathcal L_{sparse}
+
\lambda_{norm}\mathcal L_{norm}
}
\]

其中：

\[
\mathcal L_{sparse}
\]

针对 \(\Xi_q\)，并在必要时对空间矩阵 \(C\) 做 group sparsity。

\[
\mathcal L_{norm}
\]

用于 latent normalization / decorrelation。

初始版本不加入：

- energy loss；
- classification loss；
- Lyapunov loss；
- stability-boundary loss。

这些只在动力学结构稳定后加入。

---

## 10. 分阶段联合训练

### Stage A：latent warm-up

目标：先得到非退化、有 electrical information 的 latent state。

训练：

\[
\mathcal L_{dyn}
+
\mathcal L_{force}
+
\mathcal L_{elec}
+
\mathcal L_{norm}.
\]

此阶段 sparsity 较弱。

### Stage B：加入短时 rollout

加入：

\[
0.2/0.5/1\text{ s}
\]

autonomous rollout。

防止只得到 pointwise-good / rollout-bad latent coordinate。

### Stage C：逐步增强 sparsity

逐步提高 \(\lambda_{sp}\)，并周期性 threshold / prune \(\Xi_q\)。

每次剪枝后继续联合拟合：

- encoder \(E_\theta\)；
- SINDy coefficients \(\Xi_q\)；
- topology map \(C\)；
- decoder \(D_\phi\)。

### Stage D：冻结 SINDy support

当非零项集合稳定后：

\[
\operatorname{supp}(\Xi_q)
\]

冻结。

只重新拟合非零系数、encoder、topology map 与必要 decoder 参数。

### Stage E：长时 rollout refinement

逐渐加入：

\[
2/5/8\text{ s}
\]

rollout。

长时 rollout 使用 event-balanced / tail-aware sampling。

### Stage F：最终结构冻结

冻结：

- latent dimension \(r\)；
- history window；
- measurement set；
- SINDy support；
- topology map structure；
- ODE parameters。

随后才能进入 frozen test。

---

## 11. Latent dimension discovery

按：

\[
r=1\rightarrow2\rightarrow3
\]

依次实验。

每个维数重新联合训练，不通过简单 PCA 方差决定阶次。

阶数选择同时考虑：

1. pooled / mean RMSE；
2. worst-10% CVaR；
3. stable / unstable qualitative behavior；
4. SINDy 非零项数量；
5. 不同 fault groups 下 support consistency；
6. topology modes 是否稳定；
7. 长时 boundedness；
8. modal / stability geometry improvement。

优先选择：

> 能以最少 latent dimension 和最少 SINDy terms 解释主要跨扰动动力学的模型。

不因为 \(r=3\) 的训练误差更低就自动接受更高阶模型。

---

## 12. Topology mode 约束

### 12.1 归一化

为避免：

\[
Cq=(\alpha C)(q/\alpha)
\]

产生不可辨识尺度，对每个空间 mode 约束：

\[
\|C_j\|_2=1.
\]

必要时：

\[
C^TC\approx I.
\]

### 12.2 稀疏性

鼓励每个 latent mode 作用在有限 network regions，而不是全网络均匀扩散。

采用：

- edge group sparsity；
- region sparsity；
- physically admissible support mask。

### 12.3 Generator/load ports

拓扑映射必须能够区分：

- generator-generator exchange；
- generator-load power transfer；
- network-loss contribution。

不得再次强制整个 residual 具有 zero-sum generator-only structure。

---

## 13. History window 与 observability

第一轮固定：

\[
T_{obs}=100\text{ ms}
\]

避免同时改变过多因素。

结构基本成立后，再比较：

\[
50/100/200\text{ ms}.
\]

若 \([\delta,\omega]\) history 对 discovered q 的 cross-group reconstruction 明显不足，再比较：

\[
[\delta,\omega,P_e]
\]

和：

\[
[\delta,\omega,V,I].
\]

该实验回答：

> memory state 是否理论存在，以及现有 measurement set 是否足以观测，是两个不同问题。

---

## 14. Ensemble SINDy 的使用条件

第一阶段不使用 ensemble SINDy。

先使用单一 \(\Xi_q\) 验证联合发现机制是否成立。

仅在以下条件满足后启用 ensemble：

1. latent coordinate 已稳定；
2. autonomous rollout 明显优于 M3；
3. 单一 SINDy support 对 threshold 较敏感。

此时使用多个 sparsity thresholds 的 ensemble 检查：

- term selection frequency；
- coefficient sign consistency；
- group consistency；
- support robustness。

最终模型仍输出一个明确稀疏 ODE，而不是 ensemble predictor。

---

## 15. 防止联合训练作弊

必须防止以下退化解。

### 15.1 Latent collapse

禁止：

\[
q\equiv0.
\]

通过 electrical reconstruction、rollout 与 variance normalization 避免。

### 15.2 Time encoding

Encoder 不输入：

- absolute time；
- trajectory index；
- fault ID；
- duration；
- label。

防止：

\[
q\approx t
\]

等伪动力学。

### 15.3 Event identity leakage

所有 history windows 必须继承 trajectory-level group split。

来自同一事件的不同窗口不能跨 train / validation。

### 15.4 Future leakage

Encoder 的窗口终点严格为当前时刻 \(t\)，不得包含未来样本。

### 15.5 Decoder shortcut

Decoder 不允许直接读取未来 trajectory，也不允许绕过 \(q\) 使用高维 full-state 输入。

---

## 16. 评价指标

### 16.1 Prediction

统一输出：

- 0.2 s；
- 0.5 s；
- 1 s；
- 2 s；
- 5 s；
- 8 s；

的：

- pooled RMSE；
- mean RMSE；
- worst-10% CVaR；
- worst-k event；
- long-horizon boundedness。

### 16.2 Cross-disturbance

按 fault group 报告：

- group mean；
- group worst；
- unseen-group generalization；
- SINDy support consistency；
- topology mode consistency。

### 16.3 Structure

记录：

- latent dimension；
- nonzero SINDy terms；
- coefficient signs；
- coefficient magnitude；
- mode spatial support；
- mode time scales；
- parameter sensitivity。

### 16.4 Physical interpretation

使用保存的 full-order states 做后验解释：

检查 \(q_j\) 与：

- \(E'_q\)；
- \(E'_d\)；
- \(E_{fd}\)；
- AVR states；
- terminal voltage；
- \(P_e\)；
- load voltage / power；

之间的：

- correlation；
- lagged correlation；
- regression；
- subspace alignment。

这些只用于解释，不参与定义 \(q\)。

---

## 17. 验收门

### 17.1 Dynamics gate

Joint model 相比 M3 至少满足：

- pooled RMSE 明确改善；
- mean RMSE 明确改善；
- worst-10% CVaR 不出现不可接受恶化；
- 2–8 s 不新增明显非物理解发散。

### 17.2 Memory gate

要求：

1. \(q\) 非退化；
2. \(q\) 的 SINDy ODE 稀疏；
3. 同一 ODE support 服务所有训练事件；
4. unseen disturbance groups 仍有收益；
5. encoder-only latent 与 autonomous latent 不明显分叉；
6. latent mode 对 electrical correction 有稳定作用。

### 17.3 Structure gate

要求：

- \(r\le3\) 为优先范围；
- SINDy terms 数量受控；
- topology modes 有明确空间结构；
- support / sign 对不同数据子集具有稳定性；
- 不依赖 parameter boundary / clipping 才能工作。

### 17.4 Stability gate

在进入最终稳定性研究前，至少要求：

- equilibrium 正确；
- Jacobian 无非物理 unstable modes；
- modal damping 相比 M3 不恶化；
- stable / unstable free rollout 不出现系统性新误判。

冻结测试只在全部模型结构冻结后使用。

---

## 18. 能量与稳定性分析

联合 latent-SINDy 模型通过后，再进入该阶段。

目标 reduced ODE：

\[
\dot x_r=F_r(x_r).
\]

### 18.1 Jacobian

在 equilibrium：

\[
A_r=
\left.
\frac{\partial F_r}{\partial x_r}
\right|_{x_r^*}.
\]

比较：

- dominant eigenvalues；
- damping ratios；
- modal frequencies；
- full-order dominant modes；
- M3 baseline。

### 18.2 Energy/storage discovery

以 M3 已有：

\[
H_3
\]

为基础，寻找：

\[
\boxed{
H_{total}
=
H_3+H_q+H_c
}
\]

其中：

- \(H_q\)：latent electrical storage；
- \(H_c\)：mechanical-electrical/topological coupling storage。

不预设单一 global critical energy。

### 18.3 Stability boundary

优先研究：

\[
\rho_c(v)
\]

或：

\[
G(x_r)=0,
\]

表示方向相关或状态相关的 stability boundary。

Energy 作为重要 coordinate / sufficient certificate，而不是强制作为唯一全局 stability margin。

---

## 19. 推荐执行顺序

### Loop A：Joint latent-SINDy prototype

- 固定 M3 backbone；
- \(r=1\)；
- 100 ms history；
- small GRU；
- 单一 SINDy；
- topology map \(C\)；
- 1 s rollout；
- 不做 energy/stability regularization。

目标：证明 joint training 本身可行。

### Loop B：Topology / algebraic-port refinement

- 加入 generator-network-load topology；
- 加入 \(P_e,V,P_L\) electrical reconstruction；
- 检查 load-port decomposition；
- 保持 \(r=1\)。

### Loop C：Latent dimension selection

依次：

\[
r=1,2,3.
\]

按 prediction + sparsity + group consistency 选择最小有效阶次。

### Loop D：Sparse support discovery

- sparsity continuation；
- threshold pruning；
- support freeze；
- refit。

### Loop E：Long-horizon rollout

逐步扩展到：

\[
2/5/8\text{ s}.
\]

重点监控 tail 与 unstable events。

### Loop F：Observability study

对 accepted latent ODE 比较：

- 50 / 100 / 200 ms；
- \([\delta,\omega]\)；
- \([\delta,\omega,P_e]\)；
- \([\delta,\omega,V,I]\)。

### Loop G：Structure robustness

必要时启用 ensemble SINDy，仅用于 support robustness，而不是最终预测器。

### Loop H：Energy / modal / stability analysis

完成：

- equilibrium；
- Jacobian；
- modal comparison；
- storage structure；
- numerical basin；
- direction-dependent stability boundary。

### Loop I：Final freeze and prospective test

模型、support、latent dimension、observer window、measurement set、threshold 全部冻结后，才能使用全新 prospective frozen test。

---

## 20. 论文级目标模型

最终期望获得：

\[
\boxed{
\begin{aligned}
\dot\delta
&=\omega,\\
M\dot\omega
&=
p
-\mathcal E_0(\delta)
-D\omega
-r_{alg}(\delta,q)
-S_GB_{net}Cq,\\
\dot p
&=-A_mp-B_m\omega,\\
\dot q
&=
\Theta_q(\delta,\omega,p,q)\Xi_q.
\end{aligned}
}
\]

并具有：

\[
\boxed{
q_t=E_\theta(\mathcal H_t)
}
\]

作为训练期 latent coordinate generator 与预测期 initial-state observer。

核心研究贡献定位为：

1. 保留机械与摆动方程物理骨架；
2. 从短时历史中联合发现最小 latent electrical state；
3. latent state 通过真实网络拓扑作用于电气力；
4. latent dynamics 由物理候选库中的稀疏 SINDy 方程表示；
5. latent coordinate、空间模式与 ODE 联合训练，而非串行拼接；
6. future rollout 完全由显式 reduced ODE 完成；
7. 在最终 explicit ODE 上进一步分析 energy、modal structure 与 stability boundary。

最终研究问题为：

> **能否在保留电力系统已知物理骨架的前提下，通过历史编码、网络拓扑约束与 SINDy 稀疏动力学的联合优化，自动发现能够闭合高阶电气记忆的最小状态及其显式 ODE，并由此获得同时具备长期预测能力和稳定性分析价值的低阶 post-fault 模型？**
