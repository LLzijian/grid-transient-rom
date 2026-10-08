# 任务二：具有拓扑空间结构的物理约束稀疏记忆闭合与稳定性分析

## 1. 总体目标

基于统一 post-fault 系统的数据集，发现一个**低阶、显式、稀疏、具有网络拓扑空间结构和记忆闭合能力**的暂态动力学模型。

研究主线为：

\[
\boxed{
\text{M3-Pm physical backbone}
\rightarrow
\text{electrical residual decomposition}
\rightarrow
\text{topology-aware memory-state discovery}
\rightarrow
\text{sparse memory ODE}
\rightarrow
\text{history observer}
\rightarrow
\text{energy / stability structure}
}
\]

最终目标不是增加一个任意 latent state，而是回答：

> 在显式机械功率动态已经分离后，M3-Pm 尚未解释的高阶电气/控制动态，能否被少数具有明确空间作用模式和时间尺度的 memory states 闭合，并形成可用于稳定性分析的显式 reduced ODE？

---

## 2. 当前基础

以下内容作为后续工作的固定基础，不重复优化：

- Pilot 数据已完成；正式数据扩展正在进行；
- M2 已冻结，仅作为基线；
- M3-Pm 已证明具有显著长期轨迹预测收益；
- M3-Pm 仍存在稳定性漏判、模态阻尼差异和保守能量界覆盖不足；
- 已尝试的全局标量 memory、自由指数恢复、受驱动 memory、局部稳定约束及小型 GRU 均未形成可接受 M4；
- oracle hidden correction 存在预测潜力，但当前 50 ms history observer 跨扰动泛化不足。

后续将 M3-Pm 视为：

\[
\boxed{\text{physical backbone / scaffold}}
\]

而不是最终研究终点。

---

## 3. 基础动力学骨架

记：

\[
p=P_m-P_m^*.
\]

冻结的 M3-Pm 骨架为：

\[
\boxed{
\begin{aligned}
\dot\delta &= \omega,\\
M\dot\omega &= p-\mathcal E_0(\delta)-D\omega,\\
\dot p &= -A_mp-B_m\omega.
\end{aligned}}
\]

其中 \(\mathcal E_0(\delta)\) 保留网络正弦耦合结构。

后续不直接修改 M3 已确认的机械侧结构，而重点研究其剩余电气/控制误差。

---

## 4. 核心研究对象：M3 剩余电气力

利用 full-order trajectory 定义 M3 未解释的动态残差：

\[
\boxed{
r_e(t)
=
M\dot\omega_{FOM}(t)
-p(t)
+\mathcal E_0(\delta(t))
+D\omega(t)
}
\]

理想 M3 应满足：

\[
r_e(t)=0.
\]

实际的 \(r_e\) 被视为后续结构发现的直接对象。

所有 memory 设计必须回答：

> 它解释了 \(r_e\) 中哪一部分，以及该部分是否具有可重复的动态结构？

不再先假设一个 \(q\) 的具体物理形式再看能否降低 RMSE。

---

## 5. 第一阶段：瞬时结构与真正记忆的分离

将剩余项分解为：

\[
\boxed{
r_e(t)=r_{inst}(z(t))+r_{mem}(t)}
\]

其中：

\[
z=[\delta,\omega,p].
\]

### 5.1 瞬时项

\(r_{inst}\) 只允许依赖当前 reduced state，用于修正 M3 中过于简化的静态/瞬时电气关系。

候选函数优先采用网络物理结构：

\[
\sin(B^T\delta),
\qquad
\cos(B^T\delta)-\cos(B^T\delta^*),
\qquad
\omega,
\qquad
p.
\]

采用 constrained sparse regression 识别最小瞬时修正。

### 5.2 验收原则

瞬时 residual fit 不能单独作为依据。

任何 \(r_{inst}\) 必须进入 autonomous rollout 后仍有收益，否则视为 pointwise force fitting，不保留。

### 5.3 输出

得到：

\[
\boxed{
r_{mem}=r_e-r_{inst}}
\]

后续 memory-state discovery 只针对该动态剩余项进行。

---

## 6. 第二阶段：具有拓扑空间结构的 memory mode discovery

### 6.1 不再使用全系统共享倍率

不再把 memory 表示为：

\[
e^q\mathcal E(\delta)
\]

这种全系统统一缩放。

memory 应允许不同网络区域具有不同但低秩的空间响应。

### 6.2 基于网络拓扑的空间表示

令发电机网络 incidence / reduced incidence matrix 为：

\[
B_g.
\]

memory force 采用：

\[
\boxed{
r_{mem}=B_g C q}
\]

其中：

- \(q\in\mathbb R^r\)：少量 memory modes；
- \(C\)：待发现的 edge/network spatial modes；
- \(B_gCq\)：memory 对 generator electrical force 的空间分布。

第一阶段从：

\[
r=1
\]

开始，之后最多逐步测试：

\[
r=2,3.
\]

不采用“一台机一个 q”的高维结构。

### 6.3 mode selection

mode 维数由以下指标共同决定：

- residual explained variance；
- autonomous rollout RMSE；
- worst-10% CVaR；
- disturbance-group consistency；
- full-state association；
- stability-boundary improvement。

不能仅按 PCA explained variance 或训练误差选择维数。

---

## 7. 第三阶段：显式稀疏 memory ODE 发现

发现 memory state 后，再识别其动力学：

\[
\boxed{
\dot q
=
-\Lambda q
+
G\,\Phi(\delta,\omega,p,q)
}
\]

其中：

\[
\Lambda\succ0
\]

保证 memory 自身具有稳定恢复成分。

### 7.1 候选函数库

候选库保持小规模并受物理约束：

\[
\Phi=
\{
B_g^T\omega,
\sin(B_g^T\delta),
\cos(B_g^T\delta)-\cos(B_g^T\delta^*),
p,
q
\}
\]

必要时增加少量低阶 cross terms。

### 7.2 稀疏发现

采用 constrained SINDy / sequential thresholded regression / sparse regression 识别：

- \(\Lambda\)；
- driving terms；
- coupling coefficients；
- 必要的 mode interaction。

要求：

- sparse；
- stable；
- 参数跨 disturbance groups 一致；
- autonomous rollout 有收益；
- 不依赖 future trajectory 才能运行。

### 7.3 指数恢复的定位

原模型：

\[
\dot q=-a_qq
\]

保留为最简单的 null model。

若 sparse discovery 最终仍选择无驱动自由恢复，则数据支持单一 relaxation mode；若选择 driven terms，则采用数据发现的结构。

---

## 8. 优先探索的能量一致 memory 结构

在一般 sparse discovery 之外，单独检验一类具有明确储能—耗散结构的拓扑 memory：

\[
\boxed{
\begin{aligned}
M\dot\omega
&=
 p-\nabla U(\delta)-D\omega-B_gCq,\\
\dot q
&=
C^TB_g^T\omega-\Lambda q.
\end{aligned}}
\]

定义：

\[
H
=
H_3+rac12q^Tq.
\]

若 M3 已满足：

\[
\dot H_3
=
-\omega^TD\omega
-p^TB_m^{-1}A_mp,
\]

则上述结构有：

\[
\boxed{
\dot H
=
-\omega^TD\omega
-p^TB_m^{-1}A_mp
-q^T\Lambda q
\le0.
}
\]

该结构作为高优先级候选，因为：

- memory 具有明确的能量交换解释；
- spatial coupling 由网络拓扑约束；
- \(q\) 为低维 electrical storage modes；
- 可以直接支持后续稳定性分析。

如果数据明确不支持该结构，再放宽到一般 sparse driven memory。

---

## 9. 第四阶段：history observer

只有在已经得到：

1. 有明确预测贡献的 \(q\)；
2. 可 autonomous rollout 的显式 \(q\)-ODE；

之后才训练 observer。

目标：

\[
\boxed{
q_0=E_\phi(\mathcal H_{0:T_{obs}})
}
\]

### 9.1 观察窗口

比较：

\[
T_{obs}\in\{50,100,200\}\text{ ms}.
\]

先形成 information curve，再决定最终窗口。

### 9.2 输入观测集

按信息量逐级测试：

#### O1

\[
\delta,\omega
\]

#### O2

\[
\delta,\omega,V,I
\]

如果 \(\delta,\omega\) 无法稳定恢复 \(q_0\)，优先增加实际可测物理量，不优先增加神经网络容量。

### 9.3 observer complexity

按顺序：

1. linear / delay features；
2. small GRU；
3. small TCN。

网络只用于初始化一次，future rollout 完全由显式 ODE 完成。

---

## 10. M3 能量基线

保留 M3 已有储能结构：

\[
\boxed{
H_3
=
\frac12\omega^TM\omega
+U(\delta)
+\frac12p^TB_m^{-1}p
}
\]

以及：

\[
\boxed{
\dot H_3
=
-\omega^TD\omega
-p^TB_m^{-1}A_mp
\le0.
}
\]

该结果作为所有 memory extension 的物理基准。

M4 不要求 \(H_3\) 本身单调，因为 hidden electrical states 可以储存并释放能量；但新增 memory 必须给出对应的：

\[
H_q
\]

或 coupling storage interpretation。

---

## 11. 稀疏能量函数发现

accepted memory ODE 确定后，构造：

\[
\boxed{
H_4
=
H_3+H_q(q)+H_c(\delta,p,q)
}
\]

优先候选：

- \(q^TQq\)；
- topology-aware \(q\)-angle coupling；
- 少量 \(qp\)；
- 与已发现 ODE 一致的 storage terms。

使用小型稀疏 library：

\[
H_4=\Theta_H(x_r)\xi_H.
\]

验收：

1. equilibrium 附近局部正定；
2. 与显式 ODE 的能量交换一致；
3. stable trajectory 上 storage/dissipation 可解释；
4. 不通过牺牲 trajectory prediction 获得漂亮能量曲线。

完整：

\[
\dot x=[J(x)-R(x)]\nabla H
\]

仅在上述结构得到数据支持后再尝试。

---

## 12. 稳定性与稳定裕度

稳定性分析只针对最终显式 reduced ODE。

### 12.1 Local stability

计算：

\[
A=
\left.\frac{\partial F_r}{\partial x}\right|_{x^*}
\]

比较：

- dominant eigenvalues；
- damping ratios；
- modal frequencies；
- mode shapes；
- full-order dominant modes。

memory model 必须改善或至少合理解释 M3 已观察到的 modal mismatch。

### 12.2 Nonlinear boundary

不预设存在统一 scalar：

\[
H_{crit}.
\]

优先研究方向相关稳定边界：

\[
\boxed{
rho_c(v)}
\]

或一般 reduced boundary：

\[
\boxed{G(x_r)=0}.
\]

### 12.3 Energy role

能量 \(H(x_r)\) 作为：

- sufficient certificate；
- stability coordinate；
- boundary feature；
- severity ranking quantity。

只有当数据明确支持近似统一 critical energy 时，才定义：

\[
\Delta H=H_{crit}-H.
\]

---

## 13. 正式数据扩展后的第一批实验

Loop 0039 数据扩展完成后，不直接训练新 M4。

按以下顺序执行：

### A. Expanded-data QA

确认：

- 新 fault-bus groups 完成；
- post-fault system 一致；
- full-order states 完整；
- 新 TRAIN / VALIDATION / frozen TEST 无泄漏；
- 旧 test 保持 consumed，不用于调参。

### B. Electrical residual atlas

在 expanded TRAIN 上计算 \(r_e(t)\)，分析：

- 空间 rank；
- 时间尺度；
- stable / unstable 差异；
- disturbance-group consistency；
- 与 full-order states 的相关结构。

### C. Instantaneous correction

识别最小 \(r_{inst}(z)\)，必须通过 autonomous rollout 验证。

### D. Dynamic residual modes

对：

\[
r_{mem}=r_e-r_{inst}
\]

发现 topology-aware spatial modes 和对应 full-state coordinates。

### E. Memory ODE discovery

从 \(r=1\) 开始识别 stable sparse driven dynamics。

### F. Observer

只有 E 通过后才从 history 估计 \(q_0\)。

---

## 14. 验收门

### 14.1 Prediction

相对 M3-Pm：

- pooled RMSE 有明确收益或至少不恶化；
- mean RMSE 有明确收益；
- worst-10% CVaR 不出现系统性恶化；
- severe unstable events 不因平均优化被牺牲；
- 2–8 s autonomous rollout 无新非物理解发散。

### 14.2 Structure

- memory state 数量尽可能少；
- spatial modes 具有网络拓扑结构；
- ODE 显式且 sparse；
- recovery rates 稳定；
- 参数跨 disturbance groups 可复现；
- 不使用 fault metadata；
- 不使用 future leakage。

### 14.3 Physics

- memory 对 M3 residual 有明确解释；
- full-state association 可重复；
- topology coupling 不违反网络结构；
- energy/storage interpretation 自洽；
- 不依赖 clipping 或异常大参数才能工作。

### 14.4 Stability

最终模型至少要求：

- 不增加 frozen validation 上的 false-stable failure；
- modal agreement 相比 M3 有改善或明确保持；
- numerical stability boundary 更接近 full-order；
- energy / boundary relation 具有稳定趋势。

---

## 15. 模型维数扩展规则

默认从：

\[
r=1
\]

开始。

只有当 residual atlas 与 full-state analysis 同时显示第二个独立稳定模式，并且 \(r=1\) 在多个 disturbance groups 上存在一致性结构缺陷时，才增加：

\[
r=2.
\]

同理，\(r=3\) 必须由新增独立动态模式证明。

禁止仅通过训练 loss 改善增加维数。

---

## 16. 推荐执行顺序

### Step 1
完成 Loop 0039 数据扩展与 QA，冻结新的 test groups。

### Step 2
在 expanded TRAIN 上建立 electrical residual atlas：

\[
r_e(t).
\]

### Step 3
分离并验证 instantaneous electrical correction：

\[
r_{inst}(z).
\]

### Step 4
从剩余 dynamic residual 中发现 topology-aware memory spatial modes：

\[
r_{mem}=B_gCq.
\]

### Step 5
利用 full-order states 发现与这些 force modes 对应的低维 state coordinates：

\[
q=W^Th.
\]

### Step 6
发现显式 sparse memory ODE：

\[
\dot q=-\Lambda q+G\Phi(z,q).
\]

优先检验 energy-compatible topology memory。

### Step 7
用 autonomous rollout、tail、stable/unstable 和 cross-group validation 选择 \(r\) 和 ODE structure。

### Step 8
在 accepted memory dynamics 上训练 history observer，并比较 50/100/200 ms 以及不同 measurement sets。

### Step 9
在 \(H_3\) 基础上发现最小 \(H_q/H_c\)，形成完整 storage / dissipation structure。

### Step 10
进行 equilibrium、Jacobian、modal、direction-dependent boundary 和 stability-margin 分析。

### Step 11
所有模型结构、observer、threshold 和 stability definition 冻结后，只使用新的 frozen TEST 一次。

---

## 17. 目标模型

最终期望模型形式为：

\[
\boxed{
\begin{aligned}
\dot\delta
&=\omega,\\
M\dot\omega
&=
 p
-\nabla U_{sparse}(\delta)
-D\omega
-B_gCq,\\
\dot p
&=-A_mp-B_m\omega,\\
\dot q
&=-\Lambda q
+G\Phi_{sparse}(\delta,\omega,p,q).
\end{aligned}}
\]

其中：

- \(U_{sparse}\)：instantaneous electrical structure；
- \(q\)：由 full-order data 发现的少量 memory modes；
- \(C\)：具有网络拓扑含义的 spatial coupling modes；
- \(\Lambda\)：memory time scales；
- \(\Phi_{sparse}\)：由 constrained sparse discovery 选出的驱动结构；
- \(q_0\)：最终由短时实际观测估计；
- future rollout 完全由显式 ODE 完成。

该模型应进一步允许构造：

\[
H=H_3+H_q+H_c
\]

并用于 nonlinear stability geometry 与稳定裕度研究。

---

## 18. 最终研究问题

本任务最终回答：

> **在显式机械功率慢动态已经分离后，能否从高阶系统剩余电气力与 full-order hidden states 中发现少数具有网络拓扑空间结构的 memory modes，并进一步识别其稀疏、稳定、可观测的显式动力学，使 reduced model 同时提高长期预测、模态结构和非线性稳定边界的一致性？**

核心贡献应体现为：

\[
\boxed{
\text{state discovery}
+
\text{topology discovery}
+
\text{ODE discovery}
+
\text{energy/stability structure}
}
\]

而不是单纯增加一个 latent neural state。
