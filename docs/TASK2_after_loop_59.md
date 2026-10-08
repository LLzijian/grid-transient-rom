# TASK2 after Loop 0059：联合潜状态—拓扑—稀疏动力学的长时闭合与稳定性发现

## 1. 总体目标

在已验证的 M3-Pm 物理骨架上，联合发现能够由短时历史恢复、通过真实网络拓扑作用于发电机、并由显式稀疏 ODE 自主推进的低维电气记忆状态。

当前研究主线固定为：

\[
\boxed{
\text{short-history encoder}
\rightarrow
q
\rightarrow
\text{topological electrical coupling}
\rightarrow
\text{sparse ODE}
\rightarrow
\text{long autonomous rollout}
\rightarrow
\text{stability structure}
}
\]

目标不再是验证 memory 是否存在，而是回答：

> **当前 r=3 联合 memory closure 是否能够跨越 1–8 s 时间尺度形成真正闭合、稀疏、可解释的 reduced ODE，并保留稳定性相关结构？**

最终模型仍需满足：

1. \(\delta,\omega,P_m\) 保留显式物理方程；
2. latent state 仅表示 M3-Pm 未解释的电气/控制动态；
3. encoder 只用于状态初始化，不替代 future dynamics；
4. latent dynamics 必须为显式稀疏 ODE；
5. memory force 必须经过网络拓扑映射，而不是任意 generator-wise 输出；
6. TRAIN 可充分使用完整暂态过程，但 frozen test 不得用于任何结构、参数、阈值或阶数选择；
7. 预测性能、尾部误差和长期有界性优先于 latent 坐标的统计正交性。

---

## 2. 已确认的阶段性结论

### 2.1 数据与 M3-Pm

- Pilot 与后续扩展数据均已建立独立 train / validation / frozen-test 分组；
- M3-Pm 在首次 frozen test 上确认了显著轨迹预测收益；
- M3-Pm 仍存在稳定性误判、模态差异和长期电气残差，因此仅作为 physical backbone，不作为最终结构发现结果。

### 2.2 Loop 0053–0058

联合训练已经证明以下方向可行：

\[
\boxed{
\text{history encoder}
+
\text{topological latent force}
+
\text{SINDy latent ODE}
}
\]

其中 Loop 0058 是当前 development champion：

- latent dimension：\(r=3\)；
- observation window：100 ms；
- prediction 时 encoder 只调用一次；
- 1 s autonomous rollout 中 mean / pooled / worst-tail 均显著优于 M3；
- validation fault groups 全部获得预测改善；
- 三个 latent components 的移除均造成性能下降。

### 2.3 Loop 0059 latent-order audit

Loop 0059 不再使用 latent covariance orthogonality 作为状态阶数判据。

当前证据支持：

- r=3 latent subsystem 在局部线性化下具有三个可观测、可激励的动态方向；
- topology output mapping 不呈明显 rank collapse；
- 三个 dynamic modes 的删除均损害自主预测；
- r=3 向 r=2 的结构化约简会明显损害预测性能。

因此：

\[
\boxed{r=3}
\]

被保留为当前有证据支持的候选最小阶数。

但 0059 同时发现慢时间尺度和一个近积分型 mode，且 5–8 s 自主预测明显恶化，因此当前最大问题已从“阶数是否正确”转为：

> **1 s 训练是否不足以识别真实慢时间尺度 latent dynamics。**

---

## 3. 当前模型骨架

定义：

\[
p=P_m-P_m^*.
\]

当前 reduced state：

\[
\boxed{x_r=[\delta,\omega,p,q]},\qquad q\in\mathbb R^3.
\]

物理骨架：

\[
\dot\delta=\omega
\]

\[
M\dot\omega
=
p-\mathcal E_0(\delta)-D\omega-r_{alg}-r_{mem}
\]

\[
\dot p=-A_mp-B_m\omega
\]

latent memory：

\[
\boxed{
\dot q
=
\Theta_q(\delta,\omega,p,q)\Xi_q
}
\]

其中 memory force 经网络拓扑映射：

\[
\boxed{
r_{mem}=S_GB_{net}Cq
}
\]

而不是直接学习任意 10 维 generator force。

负荷功率、电压等继续作为电气代数端口与监督量，不定义为具有独立恢复 ODE 的 memory state。

---

## 4. History encoder

训练阶段使用滑动历史窗口产生 latent reference：

\[
q_t^{enc}=E_\theta(\mathcal H_t)
\]

其中默认：

\[
T_{obs}=100\text{ ms}.
\]

当前不继续搜索 encoder architecture，除非长时实验明确证明 initialization 是瓶颈。

预测阶段严格执行：

\[
q_0=E_\theta(\mathcal H_{0:T_{obs}})
\]

随后 encoder 不再调用：

\[
q(t>0)
\]

全部由显式 SINDy ODE 自主推进。

禁止：

- rollout 中重新读取真实历史；
- 使用 future trajectory；
- 输入 fault ID / fault bus / duration 等事件标签；
- 通过周期性 re-encoding 修正长期漂移。

---

## 5. TRAIN 数据使用原则

### 5.1 使用完整训练暂态

TRAIN 中所有物理有效轨迹均允许充分使用其完整真实暂态过程。

稳定事件可使用完整 5–15 s 真实震荡过程；物理失稳事件使用其实际存在的完整 prefix，直到物理失稳判据触发。

不得：

- 为失稳事件补齐未来；
- padding；
- frozen tail；
- 删除短困难事件以改善平均指标。

### 5.2 两类训练样本并行存在

后续训练必须同时使用：

#### A. 全轨迹局部窗口

从 TRAIN 每条可用轨迹的整个时间范围随机选取：

\[
t\rightarrow t+\Delta T_{local}
\]

局部窗口用于学习整个 reachable state region 上的 latent vector field 与 electrical mapping。

不再将随机采样上限固定在前 7 s；只要该事件仍存在真实物理轨迹，即可采样。

#### B. 清故障后的长自主 rollout

从固定初始历史：

\[
\mathcal H_{0:T_{obs}}
\]

只编码一次：

\[
q_0=E_\theta(\mathcal H_{0:T_{obs}})
\]

随后自主积分到目标 horizon。

该部分直接约束真实部署条件下的 long-horizon closure。

### 5.3 Event-balanced loss

长稳定轨迹具有更多时间点，但不得因此在 loss 中压倒短失稳轨迹。

先计算 event-level loss：

\[
L_i
=
\frac{1}{T_i}
\sum_{t\in i}
\ell_{i,t}
\]

再：

\[
L_{events}
=
\frac1N\sum_iL_i.
\]

必要时对 TRAIN 内 worst-event risk 增加小权重，但不得用 validation 事件决定权重。

---

## 6. 长时间尺度课程训练

当前不增加 latent dimension、不改变 topology、不重新设计 encoder。

核心实验为：

\[
\boxed{
1s\rightarrow2s\rightarrow5s\rightarrow8s
}
\]

的 autonomous rollout curriculum。

### Stage A：1 s warm start

以 Loop 0058 champion 为初始化。

保持当前已接受的：

- r=3；
- encoder；
- topology mapping；
- SINDy candidate library；
- sparsity support initialization。

不重新从随机参数开始。

### Stage B：2 s autonomous training

预测阶段只调用一次 encoder。

允许重新估计非零 SINDy coefficients，但不立即扩充候选库。

检查：

- 2 s mean / pooled / CVaR；
- SINDy support 是否变化；
- latent eigenvalues 是否稳定；
- q autonomous-vs-encoder discrepancy。

### Stage C：5 s autonomous training

若 2 s 仍有明确收益并数值稳定，则进入 5 s。

该阶段是识别 2–6 s 慢 dynamics 的核心。

重点检查：

- 原近零 latent eigenvalue 是否移动到有限负值；
- 2–6 s time constants 是否可重复；
- 是否出现长期 latent drift；
- topology modes 是否稳定；
- severe events 是否被牺牲。

### Stage D：8 s development evaluation / refinement

8 s 首先作为 development audit horizon。

只有 5 s 训练稳定后才允许少量 8 s refinement。

不得因 8 s validation 结果反复修改 architecture；若 8 s 表现差，先判断 training-horizon identifiability，而不是直接增加模型复杂度。

---

## 7. SINDy 结构发现策略

### 7.1 当前原则

不再采用“1 s 上找到 support 后永久冻结”的假设。

长时间训练允许检查 support robustness，但修改必须有证据。

首选 small physics-informed library：

\[
q,
\quad
\omega,
\quad
p,
\quad
\sin(B^T\delta),
\quad
\cos(B^T\delta)-\cos(B^T\delta^*),
\]

以及少量：

\[
q\omega,
\quad
qp,
\quad
q\sin(B^T\delta).
\]

禁止直接开放大规模任意 polynomial library。

### 7.2 Support stability

在 1 / 2 / 5 s curriculum 各阶段记录：

- active terms；
- coefficient signs；
- coefficient scale；
- group consistency；
- latent eigenvalues。

真正可信的结构应在延长 horizon 后趋于稳定，而不是每个 horizon 完全换一套方程。

### 7.3 Near-zero mode audit

对当前近积分 mode 不预设修正形式。

重点检验：

\[
\lambda\approx0
\]

是否随着更长时间训练逐步变为：

\[
\lambda<0.
\]

若出现随 horizon 稳定收敛的负 relaxation rate，则保留该结构；若始终接近 0，则再检查其是否为真实 conserved / integral-like coordinate。

不得人为添加：

\[
-aq
\]

只为消除零极点。

---

## 8. Neural ODE teacher：算法搜索上限

在不破坏主线的前提下，可建立一个小型 neural ODE teacher 作为容量上限诊断。

保持：

- r=3；
- encoder；
- M3-Pm physical backbone；
- topology mapping；
- 同一 TRAIN / validation split。

仅替换：

\[
\dot q=\Theta_q\Xi_q
\]

为：

\[
\dot q=f_{NN}(q,\delta,\omega,p).
\]

网络必须小型，不作为最终论文模型。

### 8.1 目的

比较：

\[
\text{SINDy}
\quad vs\quad
\text{small neural teacher}
\]

在 2 / 5 / 8 s 的性能差距。

### 8.2 解释

若 teacher 仅略优于 SINDy：

> 瓶颈更可能在 latent representation / measurement / topology，而不是 SINDy library。

若 teacher 显著优于 SINDy：

> 当前 r=3 表示空间具有足够预测容量，主要瓶颈在稀疏 ODE 表达能力。

此时进入：

\[
\boxed{
\text{teacher}
\rightarrow
\text{sparse equation distillation}
\rightarrow
\text{joint refinement}
}
\]

teacher 不得成为最终 accepted model。

---

## 9. Sparse equation distillation

若 neural teacher 显著胜出，从 teacher 和 joint encoder trajectories 中构造：

\[
(q,\delta,\omega,p,\dot q)
\]

样本。

使用受物理约束的 sparse regression 发现：

\[
\dot q=\Theta_q\Xi_q.
\]

优先：

1. 扩充少量物理候选项；
2. stability-aware but not stability-forced support selection；
3. multi-horizon coefficient consistency；
4. group-wise support consistency。

得到 sparse support 后重新回到 end-to-end joint training：

\[
E_\theta,
\quad C,
\quad\Xi_q
\]

共同微调。

---

## 10. Latent order 规则

当前默认：

\[
\boxed{r=3}
\]

不再使用 latent covariance、whitening 或统计正交性作为主要阶数门。

状态阶数依据：

- autonomous rollout contribution；
- topology output rank；
- local dynamic modes；
- observability；
- controllability / reachability；
- structured reduction performance。

### 10.1 不允许继续增阶的条件

在以下问题尚未完成前，不尝试 r>3：

- long-horizon curriculum；
- near-zero mode audit；
- neural teacher upper bound；
- 5–8 s topology / SINDy stability。

### 10.2 r=4 仅作 capacity probe

若 r=3 neural teacher 在长时 horizon 仍明显不足，才允许 r=4 teacher 作为 capacity upper-bound experiment。

r=4 不自动进入主模型。

---

## 11. 电气代数端口与 memory force

Loop 0048 以后，负荷功率不再作为独立 memory state。

保持：

\[
P_L=G_L|V_L|^2
\]

等代数关系。

联合模型中的 electrical decoder / auxiliary head 应继续监督：

- generator \(P_e\)；
- voltage magnitude；
- load active power；
- 必要的 internal electrical states。

长期目标是让：

\[
q\rightarrow V,P_e,P_L
\]

的物理映射逐渐进入真正的动力学/代数方程，而不是长期停留为独立辅助监督头。

但在完成 long-horizon closure 前，不同时重构整个 electrical decoder，避免混入多个新假设。

---

## 12. 训练损失

长时阶段建议：

\[
\boxed{
L
=
\lambda_{local}L_{local}
+
\lambda_{long}L_{autonomous}
+
\lambda_{dyn}L_{dyn}
+
\lambda_{elec}L_{elec}
+
\lambda_sL_{sparse}
+
\lambda_{tail}L_{tail}
}
\]

其中：

### \(L_{local}\)

来自完整 TRAIN 轨迹随机局部窗口，用于覆盖 reachable state region。

### \(L_{autonomous}\)

从固定 100 ms 初始历史开始，一次编码后自主预测 1 / 2 / 5 / 8 s。

这是后续最高优先级项。

### \(L_{dyn}\)

约束 SINDy rollout 与 encoder-induced latent trajectory 的局部一致性。

Encoder latent 不是 ground truth，因此该项不高于 physical rollout loss。

### \(L_{elec}\)

监督 topology force、\(P_e\)、电压、负荷功率等电气量。

### \(L_{sparse}\)

用于 SINDy support 压缩和稳定化。

### \(L_{tail}\)

只使用 TRAIN event-level risk；保持小权重，防止平均收益牺牲困难事件。

---

## 13. Algorithm Campaign 模式

从 Loop 0060 起，不再默认“每个小异常单独开一轮科研 loop”。

采用：

\[
\boxed{
\text{Champion}
\quad vs\quad
\text{small set of Challengers}
}
\]

当前 Champion：

> Loop 0058 r=3 joint topology-SINDy model。

第一阶段 challengers：

1. Champion + 2/5 s curriculum；
2. Champion + small richer SINDy library；
3. Champion structure + small neural ODE teacher；
4. 必要时 r=4 neural teacher，仅作 capacity probe。

统一使用同一 development benchmark。

### 13.1 Hard gates

只保留不可妥协的硬约束：

- no leakage；
- causal initialization；
- autonomous rollout；
- numerical boundedness；
- no artificial tail / sample deletion；
- frozen test 未使用；
- same post-fault physical system。

### 13.2 Performance ranking

其余指标主要用于 Pareto comparison，而不是大量一票否决门：

- 1 / 2 / 5 / 8 s mean；
- pooled RMSE；
- worst-10% CVaR；
- disturbance-group generalization；
- long-horizon boundedness；
- sparsity；
- support consistency；
- dynamic-mode consistency。

只有会改变科学结论的结构性异常才进入专项 audit。

---

## 14. 长时动力学验收

在进入最终 stability analysis 前，r=3 sparse ODE 至少应证明：

### Prediction

- 1–2 s 保留相对 M3 的明确优势；
- 5 s 不出现系统性失控；
- 8 s 至少保持可解释、bounded 的开发集预测；
- worst-tail 不因 long-horizon training 明显恶化。

### ODE structure

- SINDy support 在 2–5 s 后趋于稳定；
- coefficients 跨 fault groups 基本一致；
- dynamic eigenvalues 不依赖单一短 horizon；
- 近零 mode 得到长时数据支持或被修正为有限慢时间尺度。

### Topology

- 三个 latent modes 的实际 generator-port force mapping 保持非退化；
- 不出现新的 topology rank collapse；
- memory force 与 algebraic load-port correction 保持物理分工。

### Causality

- 只用 initial history；
- 5–8 s 中间不重新编码；
- no future truth；
- no event metadata。

---

## 15. 完整系统模态与稳定性

只有 long-horizon ODE 基本冻结后，再进入正式稳定性阶段。

### 15.1 Full reduced Jacobian

不再只分析 \(A_q\)。

对：

\[
x_r=[\delta,\omega,p,q]
\]

计算：

\[
\boxed{
J
=
\left.\frac{\partial F_r}{\partial x_r}\right|_{x^*}
}
\]

分析：

- dominant eigenvalues；
- damping ratios；
- oscillatory modes；
- latent-mechanical-electrical participation；
- 与 full-order modal trends 的对应。

### 15.2 Energy / storage

M3 的 \(H_3\) 继续作为已知 physical storage backbone。

在 accepted r=3 model 上研究：

\[
H_{total}
=
H_3+H_q+H_c.
\]

不提前强制：

\[
\dot H_3\le0
\]

因为 hidden electrical state 可以与机械侧交换储存能量。

### 15.3 Stability boundary

不预设存在单一全局：

\[
H_{crit}.
\]

优先研究：

- direction-dependent boundary；
- reduced-state numerical basin；
- local spectral margin；
- energy coordinate 与 boundary geometry 的关系。

---

## 16. Frozen test 使用规则

旧 frozen test 已被消费，不得再用于模型选择。

新的 prospective frozen test 在以下全部冻结前不得打开：

1. latent dimension；
2. observation window；
3. encoder architecture；
4. topology mapping；
5. SINDy library；
6. active support；
7. coefficients / training procedure；
8. long-horizon curriculum；
9. model selection rule；
10. stability metrics / classification rule。

最终 test 只执行一次 confirmatory evaluation。

---

## 17. 推荐执行顺序

### Phase 1：Long-horizon champion refinement

以 Loop 0058 为唯一主模型基础。

执行：

\[
1\rightarrow2\rightarrow5\rightarrow8s
\]

curriculum，并启用 full-trajectory local sampling。

### Phase 2：Teacher upper-bound experiment

保持 r=3 和 topology 不变，使用 small neural ODE 检验 long-horizon capacity upper bound。

### Phase 3：Equation expansion / distillation

若 teacher 明显优于 SINDy：

- 分析 teacher dynamics；
- 扩充少量 physically plausible SINDy terms；
- sparse distillation；
- joint end-to-end refit。

若 teacher 也不行：

优先检查：

- observation information；
- latent dimension capacity；
- topology/output structure；

而不是继续无限扩展 SINDy library。

### Phase 4：Long-horizon model selection

使用统一 1 / 2 / 5 / 8 s benchmark 和 Pareto filtering，选择最终 sparse champion。

### Phase 5：Scientific audit

只对最终 top model 做：

- dynamic-mode audit；
- observability / controllability；
- topology interpretation；
- full-state correlation / physical interpretation；
- numerical integration sensitivity。

### Phase 6：Stability / energy

在冻结 sparse ODE 上完成：

- equilibrium；
- full reduced Jacobian；
- modal comparison；
- storage / energy structure；
- numerical stability boundary；
- stability margin。

### Phase 7：Fresh frozen test

全部结构和规则冻结后，只打开一次新的 prospective frozen test。

---

## 18. 当前论文级目标模型

目标模型保持：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega
&=
p
-\mathcal E_0(\delta)
-D\omega
-r_{alg}(\delta,q)
-S_GB_{net}Cq,\\
\dot p
&=
-A_mp-B_m\omega,\\
\dot q
&=
\Theta_q(\delta,\omega,p,q)\Xi_q,
\qquad q\in\mathbb R^3.
\end{aligned}
}
\]

其中：

- \(q_0\) 由短时历史 encoder 一次性初始化；
- future dynamics 完全显式；
- \(C\) 表示 topology-aware spatial modes；
- \(\Xi_q\) 表示由联合训练和 sparse discovery 得到的 latent ODE；
- algebraic load/network response 与 dynamic memory state 分离；
- long-horizon trajectory、dynamic modes 和 stability geometry 共同决定最终结构。

最终研究问题为：

> **能否利用有限短时观测联合发现一个具有网络拓扑空间结构的最小电气记忆状态，并通过稀疏显式 ODE 在完整暂态时间尺度上自主闭合，从而得到同时具备长期预测能力、物理结构和稳定性分析能力的 post-fault reduced-order model？**
