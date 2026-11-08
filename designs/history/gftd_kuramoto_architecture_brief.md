# 架构简报：GFTD-Kuramoto v2.0 — 部分观测广义藏本网络的生成泛函自上而下结构发现

- **状态**。`DRAFT_ARCHITECTURE_V2_REQUIRES_REPAIR`。2026-11-07 用户已恢复此路线研究推进，具体 Run 由 root 按既有协议冻结与审查后释放，范围内无需逐轮用户确认。
- **作者与模型回执**：`root`（宿主持有模型回执，不冒充外部角色调用）
- **基准提交（source commit）**：`bd4db257daf6f1bc925880a996f717d401faae6b`
- **日期**。初稿 `2026-11-06`，证据与数学更正 `2026-11-07`。
- **沿用的部署/数据合同**：[kiki.md](../kiki.md)（100 ms 因果 $(\delta,\omega)$ 历史 + 末点真 $p_0$ + 冻结物理常数；原 `val/test` 关闭，旧消费 `test` 退出选择，`new FOM = 0`，`TASK4` 暂停）
- **前序架构存档**：[architecture_design_before_gftd_kuramoto_20261106](history/architecture_design_before_gftd_kuramoto_20261106.md)、[current_design_before_gftd_kuramoto_20261106](history/current_design_before_gftd_kuramoto_20261106.md)

---

2026-11-07 更正记录。六项本地单测通过，支持对应有限测试条件。原型尚缺独立动态 p、部分观测隐轨迹推断与部署核映射，指数截断区梯度、COI 度量和父模型嵌入需修订。以下保留 v2.0 候选的建模目标和公式，修订方程与验收以待产出的 `docs/gftd_repair_design_20261107.md` 冻结版本为准。更正前原始字节保存在 `.tmp/gftd_resume_20261107/original/docs/`。

2026-11-07 主线实施补充：[三步架构](../reports/gftd_mainline_20261107_architecture.md)具体化本简报的结构发现步骤。Run 1 从单 c 扩展为 K4 六边候选，完整端口消元后作有限支撑枚举；Run 2 用 beta=k*s、0<=beta<=k 的精确指数列及候选自洽幅值接通 fit/initialize；Run 3 在 train fit297 同期比较静态和调制方案，分开报告 L1、满足证明条件的 L2 和 report-only L3。该补充由实际 Scientist 配置 gpt-6-astra 给出并经独立设计审查；运行与采用仍按逐 Run 的冻结证据决定。原文中的通用接口与恢复主张不因设计完成而视为实现完成。

## 1. 科学问题与主张

1. **未解释现象与核心瓶颈**：
   - 在仅观测相位与角频率、隐藏幅值及内部控制器状态的网络中，需要区分动态耦合、静态非线性、加性记忆及初始化/优化不足。
     1. `p_constant_norm_20261031` 的六机发布必要条件空交集仅关闭固定父参数、发布数值系统下的特定常数补救支线，失配来源仍待辨别。
     2. 左零和 $1^\top Q = \lambda 1^\top$ 保留 $d(1^\top r)/dt = \kappa 1^\top\omega-\lambda1^\top r$。它约束未加权总隐力，仍可能经其他状态影响惯量加权共同转速。`memory_transport_20261029` 的 A/S/P 各组在 `train fit297` 真实可用区间均漏检 21/30 个失稳样本，8 s 均漏检 4/30、误报 0/267；历史数值用于动机，同期匹配比较用于归因。
2. **为什么重要（ICML / NeurIPS 方法论定位）**：
   - **具有惯性的广义藏本网络（Generalized Inertial Kuramoto Networks）**是物理、神经与工程同步系统的普适数学母体。若能从第一性原理出发，仅凭短窗 $(\theta,\omega)$ 观测**自上而下（Top-Down）发现有效耦合拓扑、隐变量动态调制的耦合强度与耗散门控记忆结构**，同时严格保持**单一生成泛函的能量守恒/耗散证书**，将把“单场景电力系统参数拟合”升维为可迁移、可证明的 AI4Physics 结构发现范式。
   - **高阶电力系统（IEEE 39 十机暂态）**作为含未观测磁链、励磁（AVR/PSS）、调速器（TG）及网络消元效应的非理想广义藏本真实物理系统，提供硬核实证载体。
3. **拟检验的算法贡献**。
   - GFTD-Kuramoto 用联合储能、相移不变初始化与弱形式结构选择，在给定信息与激励条件下区分静态相互作用、动态调制及加性记忆，可区分时选择简洁闭合，信息不足时报告等价类。
4. **近邻定位**。
   - 弱形式稀疏辨识、图发现、潜变量端口-哈密顿与历史编码均有相关工作。差异须围绕隐藏状态下正确字典如何得到、结构如何可辨识、初始化与部署如何一致逐项验证。独立时变边权会在所选储能导数中引入额外项，是否符号不定或破坏长期预测须具体检查。

---

## 2. 一个连贯的学习问题

### 2.1 五个一级对象定义

- **合法观测 $\mathcal{H}_0$ 与预测 $y$**：
  - 部署输入仅限 $[-T_{\text{hist}}, 0]$（$T_{\text{hist}} = 100\text{ ms}$，11 点）上的因果相位与角频率历史 $\{\theta(t_k), \omega(t_k)\}_{k=0}^K$、末点边界值 $p_0$ 及冻结惯量/阻尼常数 $(M, D)$；
  - 待预测输出为自主推进区间 $t \in (0, T_{\text{pred}}]$ 上的轨迹 $y(t) = (\tilde{\theta}(t), \omega(t), p(t))$、同步/失稳分类及 L2 吸引域裕度 $m(x_0)$。
- **状态空间 $x = (\tilde{\theta}, \omega, z^{\text{amp}}, z^{\text{mem}})$**：
  - 共同转速 $\bar{\omega}=\mathbf{1}^\top M\omega/(\mathbf{1}^\top M\mathbf{1})$ 完整保留。原 $P_{\text{sync}}^\top\theta$ 在异质惯量下不能消去共同平移，应在展开角度或局部相位图域中选一致的相对坐标。具体基与共轭度量在修订候选中冻结。
  - 隐变量按建模用途分两类，物理对应由证据判断。
    1. **调制隐变量 $z^{\text{amp}} \in \mathbb{R}^N$**。直接进入边强度，其与真实模长或暂态电势的关系待验证；
    2. **端口/恢复记忆状态 $z^{\text{mem}} = (z_{\text{coi}}, z_{\text{sync}}, p) \in \mathbb{R}^{d_m}$**：负责非马尔可夫相位滞后恢复力与共同频率/机械功率闭合。
- **群不变因果初始化器 $\mathcal{I}_\phi(\mathcal{H}_0)$**：
  - 在历史窗 $[-T_{\text{hist}}, 0]$ 上，仅提取 $\mathrm{SO}(2)$ 全局相移不变特征 $\xi(t_k) = (\bar{\omega}(t_k), \omega(t_k) - \bar{\omega}(t_k)\mathbf{1}, \sin(B_{\text{cand}}^\top \theta(t_k)), \cos(B_{\text{cand}}^\top \theta(t_k)))$；
  - 拟由因果 Graph-GRU 生成隐初态，历史与未来使用同一结构，并明确保持 $p(0)=p_0$。历史弱形式相容性需要单独检验，现有编码器与 fit 尚未接通完整推断。

### 2.2 单一联合生成泛函与自洽场方程

在完全候选图 $\mathcal{E}_{\text{cand}} = \{(i,j): 1 \le i < j \le N\}$ 上，引入**层级结构门张量** $\mathcal{S} = (S^{\text{edge}}, S^{\text{mod}}, S^{\text{mem}}) \in [0,1]^{|\mathcal{E}_{\text{cand}}| \times 2 + d_m}$，满足层级掩码约束：
$$0 \le S_e^{\text{mod}} \le S_e^{\text{edge}} \le 1 \quad (\forall e \in \mathcal{E}_{\text{cand}})$$
即：**若候选边 $e$ 不存在（$S_e^{\text{edge}}=0$），其隐变量调制门 $S_e^{\text{mod}}$ 自动归零**。

1. **联合储能泛函 $\mathcal{H}_{\mathcal{S}}(\theta, \omega, z^{\text{amp}}, z^{\text{mem}})$**：
   $$\mathcal{H}_{\mathcal{S}} = \frac{1}{2}\omega^\top M \omega + \sum_{e=(i,j)\in\mathcal{E}_{\text{cand}}} S_e^{\text{edge}} k_e \, \Phi_e\big(z_i^{\text{amp}}, z_j^{\text{amp}}; S_e^{\text{mod}}\big) \, \psi(\theta_i - \theta_j; \eta_e) + \frac{1}{2}(z^{\text{amp}})^\top \Lambda_a z^{\text{amp}} + \frac{1}{2}(z^{\text{mem}})^\top \Lambda_m z^{\text{mem}}$$
   - **中心化藏本势**：$\psi(\phi_e; \eta_e) = \cos\eta_e - \cos\phi_e - \sin\eta_e(\phi_e - \eta_e)$，在 $\phi_e = \eta_e$ 处满足 $\psi(\eta_e;\eta_e)=0, \psi'(\eta_e;\eta_e)=0$；
     该势一般不在环面上单值，需明确展开角度或局部图域，尤其调制开启后 $\psi$ 进入隐变量动力学。
   - **正定隐幅值耦合调制核**：
     $$\Phi_e(z_i^{\text{amp}}, z_j^{\text{amp}}; S_e^{\text{mod}}) = 1 + S_e^{\text{mod}} \Big( \exp\big(\gamma_i z_i^{\text{amp}} + \gamma_j z_j^{\text{amp}}\big) - 1 \Big) > 0$$
     - 当 $S_e^{\text{mod}} = 0$ 时，$\Phi_e \equiv 1$，精确退化为静态藏本耦合；
     - 当 $S_e^{\text{mod}} = 1$ 时，$\Phi_e = \exp(\gamma_i z_i^{\text{amp}} + \gamma_j z_j^{\text{amp}}) > 0$，保证有效耦合强度 $K_e^{\text{eff}}(z) = S_e^{\text{edge}} k_e \Phi_e$ 严格非负且光滑反映隐幅值涨落。

2. **统一耗散端口-哈密顿动力学 $\dot{x} = (\mathcal{J}_{\mathcal{S}} - \mathcal{R}_{\mathcal{S}}(x))\nabla \mathcal{H}_{\mathcal{S}}(x)$**：
   $$\begin{aligned}
   \dot{\theta} &= \omega \\
   M\dot{\omega} &= -\nabla_\theta \mathcal{H}_{\mathcal{S}} - D\omega - J_{\omega a}(\mathcal{S})\nabla_{z^{\text{amp}}}\mathcal{H}_{\mathcal{S}} - J_{\omega m}(\mathcal{S})\nabla_{z^{\text{mem}}}\mathcal{H}_{\mathcal{S}} \\
   \dot{z}^{\text{amp}} &= J_{\omega a}^\top(\mathcal{S})\omega - R_a(\theta,\omega,z)\nabla_{z^{\text{amp}}}\mathcal{H}_{\mathcal{S}} \\
   \dot{z}^{\text{mem}} &= J_{\omega m}^\top(\mathcal{S})\omega - \big(J_{mm}(\mathcal{S}) + R_m(\theta,\omega,z)\big)\nabla_{z^{\text{mem}}}\mathcal{H}_{\mathcal{S}}
   \end{aligned}$$
   上式保留力/速度形式的候选表示。原 $\nabla_\omega\mathcal{H}=\omega$ 的写法已更正，实际为 $M\omega$，完整反对称结构须使用一致的共轭坐标与度量。
   其中两个核心偏导给出物理自洽的**双向耦合-反作用机制**：
   - **相角恢复力（含动态耦合强度）**：
     $$\nabla_{\theta_i}\mathcal{H}_{\mathcal{S}} = \sum_{e=(i,j)} B_{ie} \, \underbrace{S_e^{\text{edge}} k_e \Phi_e(z_i^{\text{amp}}, z_j^{\text{amp}}; S_e^{\text{mod}})}_{K_e^{\text{eff}}(z^{\text{amp}})} \big(\sin(\theta_i-\theta_j) - \sin\eta_e\big)$$
   - **隐幅值恢复力与相角拉开压降反作用（Back-reaction）**：
     $$\nabla_{z_i^{\text{amp}}}\mathcal{H}_{\mathcal{S}} = \Lambda_{a,i} z_i^{\text{amp}} + \sum_{e \ni i} S_e^{\text{edge}} k_e \frac{\partial \Phi_e}{\partial z_i^{\text{amp}}} \psi(\theta_i-\theta_j; \eta_e)$$
     当 $\psi>0$ 时反作用的方向还取决于调制核导数和动力学算子，其幅值压降解释须通过对照检验。

3. **连续耗散门控算子（Physics-Aligned Continuous GRU）与 COI 解耦**：
   - 保留原候选的投影约定 $P_{\text{coi}}=M\mathbf{1}\mathbf{1}^\top/(\mathbf{1}^\top M\mathbf{1})$、$P_{\text{sync}}=I-P_{\text{coi}}^\top$，用于定位待修订的耗散矩阵：
     $$R_a(\theta,\omega,z) = \operatorname{diag}\big(\tau_a^{-1} \odot \sigma(g_a(\xi, z))\big) \succ 0, \quad R_m(\theta,\omega,z) = P_{\text{coi}}^\top \lambda_{\text{coi}}\sigma(g_{\text{coi}}) P_{\text{coi}} + P_{\text{sync}}^\top R_{\text{sync}}(\mathcal{S}, \sigma(g_{\text{sync}})) P_{\text{sync}} \succeq 0$$
   - 上述 $P$ 形式仅保留为待修订候选。异质惯量下半正定性不足以证明共同与差分动力学解耦，完整生成算子的交叉块须独立检查。

4. **完整功率账（Power Balance Identity）**：
   在一致坐标、实际 RHS 使用同一光滑储能真实梯度、$\mathcal{J}^\top=-\mathcal{J}$、$J_{mm}^\top=-J_{mm}$ 且耗散半正定的条件下：
   $$\frac{\mathrm{d}\mathcal{H}_{\mathcal{S}}}{\mathrm{d}t} = (\nabla \mathcal{H}_{\mathcal{S}})^\top (\mathcal{J}_{\mathcal{S}} - \mathcal{R}_{\mathcal{S}})\nabla \mathcal{H}_{\mathcal{S}} = -\omega^\top D \omega - (\nabla_{z^{\text{amp}}}\mathcal{H}_{\mathcal{S}})^\top R_a (\nabla_{z^{\text{amp}}}\mathcal{H}_{\mathcal{S}}) - (\nabla_{z^{\text{mem}}}\mathcal{H}_{\mathcal{S}})^\top R_m (\nabla_{z^{\text{mem}}}\mathcal{H}_{\mathcal{S}}) \le 0$$
   此为所选 ROM 的条件恒等式。现有代码指数截断区导数不相容，需要用独立储能方向差分核验。储能耗散和结构恢复、L2 吸引域分别验收。

5. **精确退化极限（Exact Embedding Limit）**：
   - $S^{\text{mod}}=0$ 只保证耦合不依赖调制隐变量。旧 40 维基线还含独立动态 p，修订须给状态/参数映射并检验全部 RHS、储能、合法初态和多步轨迹，现有单测尚未完成此证明。

---

## 3. 自上而下结构发现草案与条件命题

### 3.1 三阶段自上而下发现算法（Top-Down Functional Collapse）

1. **Stage A — 弱形式积分投影筛选（Weak-Form Variational Screening）**：
   - 给定紧支光滑测试基函数 $\varphi_m(t) \in C_c^\infty([t_a, t_b])$，对动量方程做分部积分消去 $\dot{\omega}$：
     $$-\int_{t_a}^{t_b} M \omega(t) \dot{\varphi}_m(t)\,\mathrm{d}t + \int_{t_a}^{t_b} D \omega(t) \varphi_m(t)\,\mathrm{d}t = \int_{t_a}^{t_b} \big(-\nabla_\theta \mathcal{H}_{\mathcal{S}} - J_{\omega z}\nabla_z \mathcal{H}_{\mathcal{S}}\big)\varphi_m(t)\,\mathrm{d}t$$
   - 字典必须说明隐轨迹如何仅由合法信息推断，记忆力如何进入目标，线性交叉项如何映射到部署核。秩与条件数用于诊断，不能单独推出结构可恢复或删除困难样本。
2. **Stage B — 层级结构门稀疏坍缩（Hierarchical Gate Collapse）**：
   - 在满足 $0 \le S_e^{\text{mod}} \le S_e^{\text{edge}} \le 1$ 下，求解结构风险最小化：
     $$\min_{\mathcal{S}, \theta, \phi} \; \mathcal{L}_{\text{traj}}(\mathcal{S},\theta,\phi) + \alpha_{\text{weak}} \mathcal{L}_{\text{weak}}(\mathcal{S},\theta,\phi) + \beta_{\text{edge}}\|S^{\text{edge}}\|_1 + \beta_{\text{mod}}\|S^{\text{mod}}\|_1 + \beta_{\text{mem}}\|S^{\text{mem}}\|_1$$
3. **Stage C — 规范标准型投影与精化（Canonical Gauge Projection & Refinement）**：
   - 在模型族闭合的规范变换下固定尺度，保留残余等价性。固定支撑后用同一自治 ODE 精化参数，L2 另行证明。

### 3.2 三项条件命题的更正（2026-11-07）

- **命题 1（隐状态规范等价类与规范标准型）**：
  在 $M\dot\omega=-J\nabla_zH+\cdots$、$\dot z=J^\top\omega-R\nabla_zH$ 的约定下，$\tilde z=Tz$ 要求 $\tilde J=JT^\top$、$\tilde R=TRT^\top$、$\tilde\Lambda=T^{-\top}\Lambda T^{-1}$，整个非线性储能、隐状态内部交换算子、门控及初始化均须作一致坐标变换。任意混合节点的 $T_a$ 可能破坏端点局域核，允许变换须保持模型族闭合。固定 $\Lambda=I$ 和耗散对角化仍保留符号、排列和重根旋转等价性。
- **命题 2（动态调制耦合相对静态拓扑的可辨识条件）**：
  已知、正确、固定的无噪线性字典满列秩可保证系数唯一。部分观测下还须证明隐状态可观测性、参数等价类及机制可分离性。$\gamma=0$ 时调制门不可区分，连续边门与权重乘积存在缩放冗余。恢复定理待在明确观测、激励、噪声及模型条件下重建。
- **命题 3（动态耦合下的联合 Lyapunov 吸引域内估计）**：
  正定局部 Hessian 和 $\dot H\le0$ 尚不足以推出吸引。须在明确相位图域内验证目标平衡、正定储能、解的完备性、紧致前向不变的连通子水平分支，且零耗散集合内最大不变集只有目标平衡。满足这些条件才称 ROM 吸引域内估计，$m>0$ 还须验证所属分支，$m\le0$ 保留未获证含义。

---

## 4. 竞争解释与可失败预测

| 竞争解释 | 可区分预测（Discriminating Prediction） | 使本主张失败或未决的结果 |
|---|---|---|
| **H0（动态调制改善闭合）** | 固定初始化和共同模态算子，比较调制开/关的机制预测及 L1 效应 | 无收益或混合结果按充分性、探针和区间精度判断；秩亏时先查信息和等价类 |
| **H1（静态耦合或加性记忆已充分）** | 在匹配适配机会下可达到预登记的实际精度范围 | 实践等效和机制排除各用对应证据；必要时加入静态高次谐波或加性记忆对照 |
| **H2（初始化/优化不足）** | 固定旧动力学，仅改合法初始化或给有界合理适配，改变失配 | 该解释仍在竞争；总隐力恒等式和固定参数下常数空交集不能排除一般初始化/优化解释 |
| **H3（独立时变边权）** | 去掉联合储能反作用后，预测和功率账如何变化 | 额外功率项可能符号不定，须测实际效应；耗散证明缺失不自动意味着长期发散 |

---

## 5. 下一最有价值的问题与双层推进路线

1. **Stage I（受控第一性原理广义藏本结构恢复探针 — `probe` / `discrimination`）**：
   - **目的**。先做同族静态/调制正负校准，再检验独立 Stuart-Landau 或塑性边权生成机制。真实图与隐轨迹仅供隔离的评价端使用，推断仅用协议冻结的观测。预登记观测窗口、激励、独立生成单位、恢复/误检指标和有限工作量。
   - **执行边界**。用户 2026-11-07 已恢复此路线，人工结构恢复 Run 仍须实际 Scientist 设计、协议冻结和独立审查后由 root 释放。零新增 FOM 与运行授权分别核对，成本按实际记录。
2. **Stage II（IEEE 39 真实电力系统暂态结构发现与验证 — `adaptation` / `discrimination`）**：
   - **目的**。修订候选通过相关资格、受控证据足以支持对应辨别且具体协议释放后，在已暴露 `train fit297` 上做条件内匹配比较，分开初始化、共同模态和调制作用。
     1. **L1**。沿用轨迹、共同转速 RMSE（rad/s）、归一化 p RMSE、8 s 和真实可用区间分类、原分母及数值守卫。
     2. **L2**。按完整证明条件检验合法初态所在连通分支。
     3. **L3**。有效图与 Kron 图及模态对应保持 report-only。

---

## 6. 决策与 ICML / NeurIPS 主张台账

- **2026-11-07 路线状态**。用户已恢复按评估方案研究推进。保留核心方向，先完成有界数学/代码修订和相关资格，具体新科学 Run 按既有流程冻结并审查后释放。运行事实与模型指针由 [current_state](current_state.md) 记录。

| 核心主张（Claim） | 最强已有证据 | 最强反例 / 混淆 | 仍缺的决定性证据 |
|---|---|---|---|
| **C1（有界资格）联合储能、真实梯度与父模型嵌入一致** | 2026-11-07 的 5N 修订、43 项冻结测试及明确父映射；详见资格回执 | 原 v2.0 缺独立 p 和截断梯度问题已修正；有限检查不证明全域数值稳定，$\Phi>0$ 不保证 Hessian 正定 | 两图补充梯度/功率检查已通过，逐项残差单独保存；L2 另行证明 |
| **C2（待检验）共同模态闭合可改善预测** | 总隐力约束已有解析式 | 半正定投影未证明动力学解耦；初始化、共同参考和优化解释仍在 | 一致坐标/度量推导及固定其他因素的同期匹配比较 |
| **C3（待检验）部分观测结构选择能区分机制** | 固定字典的有限回归单测、弱形式积分恒等式 | 隐轨迹未知，门/权重尺度和模型失配，线性筛选与部署核未接通 | 真值隔离的正负校准、跨机制辨别及范围相符确认 |
