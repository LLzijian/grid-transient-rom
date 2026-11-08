# 一级算法设计：部分观测广义藏本网络的自上而下结构发现

本文件定义 GFTD-Kuramoto v2.0 的架构草案。2026-11-07 用户已恢复此路线的研究推进，具体 Run 由 root 按既有协议冻结与审查后释放，范围内无需逐轮用户确认。
详细数学推导与主张台账见 [GFTD-Kuramoto 架构简报](gftd_kuramoto_architecture_brief.md)；
前序版本已归档于 [history/architecture_design_before_gftd_kuramoto_20261106](history/architecture_design_before_gftd_kuramoto_20261106.md)。
2026-11-07 更正了数学和实现主张，原始字节保存在 `.tmp/gftd_resume_20261107/original/docs/`。当前采用的方程与资格范围见 [5N 修订设计](gftd_repair_design_20261107.md)，经独立设计与实现审查；下文 v2.0 公式用于说明设计空间，不覆盖该修订的明确机械功率端口和坐标合同。

2026-11-07 主线实施补充采用[未知拓扑→动态调制→IEEE39 三步设计](../reports/gftd_mainline_20261107_architecture.md)。先用完整端口弱形式字典恢复 K4 的未知非负边权，再以 beta=k*s 与精确指数列建立层级筛选，并在每次候选改变后重推幅值。通用 fit/initialize 经人工资格后进入 train fit297 同期比较；绝对激励及逐边歧义在拓扑 Run 内判断。这个实施顺序不改变 5N 方程、部署信息或数据边界。

## 1. 普适问题定义：从部分观测发现广义藏本网络结构

本研究的算法本体针对**具有惯性与耗散的广义藏本网络（Generalized Inertial Kuramoto Networks）在部分观测下的结构发现与低阶暂态闭合**；高阶电力系统（IEEE 39 十机暂态）作为含未观测内部电磁、励磁与调速动态的**非理想真实物理验证场景**。

| 一级对象 | 数学定义与物理内涵 | 严禁出现的退化误区 |
|---|---|---|
| **信息与状态** | 部署仅用短窗因果观测 $\mathcal{H}_0=\{\theta(t_k),\omega(t_k)\}_{k=0}^K$ 与边界初值 $p_0$；状态 $x=(\tilde{\theta},\omega,z^{\text{amp}},z^{\text{mem}})$ | 借未来真值、事件标签或手工逐事件调参掩盖部分观测 |
| **假说空间** | 完全候选图 $\mathcal{E}_{\text{cand}}=K_N$ 上的层级结构门 $\mathcal{S}=(S^{\text{edge}},S^{\text{mod}},S^{\text{mem}})$，满足 $0\le S_e^{\text{mod}}\le S_e^{\text{edge}}\le 1$ | 把规定的图、边族或节点局域核称为数据发现 |
| **学习原则** | 弱形式积分投影筛选 + 层级结构门稀疏坍缩 + 规范标准型去偏精化 | 仅靠黑箱前向 ODE 伴随梯度硬训，无显式结构发现算子 |
| **可辨识性** | 在模型族闭合的坐标变换下讨论等价类，分别分析观测、隐状态推断与弱形式字典 | 把固定已知字典满秩当作部分观测结构恢复定理 |
| **判别预测** | 动态调制、静态非线性与加性记忆能否区分，共同频率闭合能否改善 | 用单一 risk 或耗散恒等式替代机制证据及 L2 证明 |

## 2. 第一性原理与三大数学物理支柱

Mori-Zwanzig 约化提示未观测作用可表现为记忆。本方案选择有限维隐状态、正弦边族和联合储能作为待检验的闭合先验，下列公式保留 v2.0 候选的设计含义。

### 2.1 隐变量纳入耦合强度与联合势能反作用定律
定义带层级结构门的总哈密顿泛函：

$$\mathcal{H}_{\mathcal{S}}(\theta,\omega,z) = \frac{1}{2}\omega^\top M\omega + \sum_{e=(i,j)\in\mathcal{E}_{\text{cand}}} S_e^{\text{edge}} k_e \, \Phi_e(z_i^{\text{amp}},z_j^{\text{amp}}; S_e^{\text{mod}})\,\psi(\theta_i-\theta_j;\eta_e) + \frac{1}{2}\|z^{\text{amp}}\|_{\Lambda_a}^2 + \frac{1}{2}\|z^{\text{mem}}\|_{\Lambda_m}^2$$

其中 $\psi(\phi_e;\eta_e)=\cos\eta_e-\cos\phi_e-\sin\eta_e(\phi_e-\eta_e)$ 为中心化藏本势，适用于明确的展开角度或局部相位图域。$\Phi_e = 1 + S_e^{\text{mod}}(e^{\gamma_i z_i^{\text{amp}}+\gamma_j z_j^{\text{amp}}}-1) > 0$ 是所选调制核。对该储能取真实梯度给出：
1. **动态有效耦合强度**：$-\nabla_{\theta_i}\mathcal{H}_{\mathcal{S}}$ 中自然涌现受隐变量调制的耦合强度 $K_e^{\text{eff}}(z)=S_e^{\text{edge}}k_e\Phi_e(z^{\text{amp}})$；
2. **相角对隐变量的反作用**：$-\nabla_{z_i^{\text{amp}}}\mathcal{H}_{\mathcal{S}} = -\Lambda_{a,i}z_i^{\text{amp}} - \sum_{e\ni i} S_e^{\text{edge}}k_e \frac{\partial\Phi_e}{\partial z_i^{\text{amp}}}\psi(\phi_e;\eta_e)$。它是否解释幅值跌落和失步，须由机制对照检验；
3. **条件功率恒等式**。坐标与度量一致、实际 RHS 使用同一光滑储能的真实梯度、$\mathcal{J}^\top=-\mathcal{J}$ 且 $\mathcal{R}\succeq0$ 时，$\dot{\mathcal{H}}=-(\nabla\mathcal{H})^\top\mathcal{R}\nabla\mathcal{H}\le0$。原 v2.0 指数截断与梯度不一致已在 5N 修订中修正；L2 另须验证平衡、紧致前向不变分支与 LaSalle 条件。

### 2.2 群不变因果 GRU 历史编码与连续耗散门控记忆
1. **因果历史编码 $\mathcal{I}_\phi(\mathcal{H}_0)$**：在 100 ms 历史窗上仅输入 $\mathrm{SO}(2)$ 全局相移不变特征 $(\bar{\omega}, \omega-\bar{\omega}\mathbf{1}, \sin(B_{\text{cand}}^\top\theta), \cos(B_{\text{cand}}^\top\theta))$，经 Graph-GRU 与历史弱形式相容约束生成初态 $z_0$。
2. **连续耗散门控算子**。正定门控提供所选 ROM 的耗散先验，时间尺度与真实系统适合性须检验。
3. **共同模态闭合候选**。共同转速采用惯量加权定义。现有投影仅保证所构造矩阵半正定，异质惯量下动力学解耦及与父模型的一致性待重推。

### 2.3 自上而下三阶段结构坍缩（Top-Down Functional Collapse）
1. **Stage A（弱形式积分投影）**：用紧支测试函数 $\varphi_m(t)\in C_c^\infty$ 分部积分消去 $\dot{\omega}$ 差分噪声，在积分域快速评估候选边与调制门的 Gram 矩阵秩及残差贡献；
2. **Stage B（层级结构门坍缩）**：在 $0\le S_e^{\text{mod}}\le S_e^{\text{edge}}\le 1$ 约束下联合稀疏化拓扑门、隐幅值调制门与记忆通道门；
3. **Stage C（坐标规范与精化）**。在保持模型族和整个非线性方程一致的变换下固定尺度，报告残余等价性，固定支撑后精化参数。L2 证书按独立证明条件检验。

## 3. 双层科学验证体系（Stage I 广义藏本 $\to$ Stage II 真实电力系统）

| 验证阶段 | 物理对象与数据边界 | 待发现的结构目标 | 核心判别证据 |
|---|---|---|---|
| **Stage I：受控广义藏本基准** | 先同族静态/调制正负校准，再用独立 Stuart-Landau 或塑性网络生成机制检验，真值与推断隔离 | 检验有效边和调制是否可区分，报告恢复率、误检与等价类 | 观测/激励条件、F1、条件数与不确定性；功率和 L2 按各自验收 |
| **Stage II：IEEE39 条件内辨别** | 已暴露的 `train fit297`，100 ms 因果合同，`new FOM = 0` | 匹配对照，分别检验初始化、共同模态与动态调制作用 | L1 沿用原分母与数值守卫，L2 须完整证明，L3 Kron/模态对应为 report-only |

## 4. 评议、嵌入极限与顶会主张准则

- **精确退化目标**。关闭调制只能证明静态耦合极限。既有 40 维父模型还需独立动态 p、状态/参数映射、功率、全部 RHS、初始化和多步轨迹一致性，2026-11-07 的 5N 修订已在两组固定人工父模型上完成这些比较，见资格回执；不外推到其他父 Q 变体。
- **角色与复核**：Scientist（`gpt-6-astra`）负责新 Run 的有界判别设计，Modeler/Experimenter（`gpt-6.1-sol`）负责冻结实现与执行，Reviewer 独立挑战竞争解释与可辨识边界，root 作最终三决定。
- 参考评审标准：[ICML 2026](https://icml.cc/Conferences/2026/ReviewerInstructions)、[NeurIPS 2026](https://neurips.cc/Conferences/2026/ReviewerGuidelines)。
