# 物理结构发现的研究定义与近期会议文献

日期：2026-10-17。用途：为Kiki后续研究方向提供文献与设计讨论。范围：ICML 2025–2026、NeurIPS 2024–2025的相关正式论文，并单列TMLR 2026补充工作。本文不改变有效实验协议、预算、模型身份或test权限。

当前项目已经建立了结构化动力学、因果初始化和自主预测的工程基础。要使“物理结构发现”成为核心贡献，需要进一步明确一个训练前未知、能由数据选择、能被独立证据检验的结构对象。当前0102–0106主要研究固定支持下的系数与初始化优化；0098曾执行有限模板库内的结构增长，二者应分别评价。

## 结构发现的操作性定义

下面是结合所列论文形成的项目分析尺度，不是ICML或NeurIPS统一发布的定义。

| 任务 | 训练前给定 | 数据决定 | 合适的主要证据 |
|---|---|---|---|
| 参数辨识 | 方程项、状态、连接和函数形式 | 系数、初态 | 预测、参数误差与数值可靠性 |
| 库内结构选择 | 候选机制或函数库 | 哪些项存在、哪些项删除 | 支持恢复、选择稳定性和必要性对照 |
| 方程形式发现 | 变量、运算规则及物理约束 | 表达式及其组合 | 等价公式恢复、结构误差和跨条件检验 |
| 对称或守恒结构发现 | 候选变换或守恒量函数族 | 对称子群、生成元或守恒量 | 生成元空间、群结构或守恒关系恢复 |
| 状态与关系发现 | 观测及观测生成假设 | 潜变量组织和交互关系 | 条件性可辨识、结构恢复及独立机制检验 |
| 证明函数发现 | 动力学方程及验证区域 | Lyapunov函数等数学证书 | 反例搜索和正式验证 |

有先验并不取消发现。关键是先验留下了哪些非平凡选择，以及所得结论是否被证据区分于其他候选。连续优化也可能发现结构：Noether’s Razor学习二次型系数，但这些系数决定未知守恒量和对称生成元。相反，连续调整已经确定的阻尼系数，通常属于参数辨识。

可将区别写为：固定结构S时优化参数theta，是参数辨识；当S本身包含待确定的项集合、函数形式、机制分解或关系，且学习与评价均针对S，才形成相应范围内的结构发现。结构变量不一定是离散变量。

## 最相关的方法论文

### DoLQ 将方程搜索与系数优化分开

Song、Shin、Lee，*Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation*，ICML 2026主会。[正式记录](https://proceedings.mlr.press/v306/song26x.html)；[作者全文](https://arxiv.org/html/2605.07323v2)。

已知状态观测和系统文字描述；Sampler提出候选方程项，数值优化器拟合参数，Scientist结合逐项评价与物理语义反馈修改结构。评价同时检查轨迹误差与真实符号项恢复。发现对象是函数形式和项组合。文字描述及预训练知识构成先验，LLM判断不等于物理证明；低维基准和噪声表现不能直接外推到部分观测电网。重点阅读图1、方法第3节、结构比较第4.4节。

### ABL-PDE 在逻辑约束下生成并选择表达式

Gao、Ge、Jiang、Zhou，*Discovering Symbolic Partial Differential Equation by Abductive Learning*，NeurIPS 2025主会。[正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/52ec5b595a8cf66afee8cfae352b3c5c-Abstract-Conference.html)。

预设变量、运算语法和复杂度范围；逻辑知识库逐渐生成表达式，选择方程项集合，神经模块提供场与导数估计。它区分项集合F与系数xi，分别检验结构恢复和系数误差。实验覆盖Burgers、Schrödinger和Navier–Stokes的不同噪声设置。当前方法主要估计各项外部的线性系数，项内部未知非线性参数仍受限。重点阅读方法第3–4节及第6.3节。

### DIF 发现跨环境共享的动力学

Gui、Li、Ji，*Discovering Physics Laws of Dynamical Systems via Invariant Function Learning*，ICML 2025主会。[正式记录](https://proceedings.mlr.press/v267/gui25a.html)；[作者全文](https://arxiv.org/html/2502.04495v2)。

预设多环境生成关系，学习将共享动力学与环境特有作用分离。例如从带不同阻尼或驱动的摆中提取共同的恢复力。它用去掉环境作用的模拟真值检验不变动力学，再以符号回归解释函数；符号表达不总是精确恢复。对项目的直接启发是识别跨工况机制，适用前提是确有机制变化的环境及合理分解假设；不同故障初态不能自动视为不同机制环境。重点阅读第5.2、5.5–5.6节。

### LieFlow 直接学习未知对称群

Chen、Park、Eijkelboom、Yang、van de Meent、Wong、Walters，*Discovering Symmetry Groups with Flow Matching*，ICML 2026主会。[正式记录](https://proceedings.mlr.press/v306/chen26dg.html)；[作者全文](https://arxiv.org/html/2512.20043v3)。

预先选择较大的候选群G，在群上学习分布，以其支撑识别对称子群H。连续和离散对称性均可作为未知对象。实验检查群元素与真实对称结构的对应，涉及点云、ModelNet10和动作数据。它展示了“发现对称性”的具体含义，但仍受候选群与识别条件约束；数据分布的对称性也不自动等同动力学方程的对称性。

### Noether’s Razor 联合学习守恒量与动力学

van der Ouderaa、van der Wilk、de Haan，*Noether’s Razor: Learning Conserved Quantities*，NeurIPS 2024主会。[正式记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html)。

预设Hamiltonian系统和二次守恒量函数族，联合学习Hamiltonian以及守恒量的二次型矩阵、线性项。近似贝叶斯模型选择处理退化常数解。除预测外，还检查学习生成元所张成空间的维数与方向。该工作说明系数学习也可以承载结构发现，前提是系数代表的未知科学对象得到验证。振子与多体保守系统的结论不能直接用于存在耗散、供能和隐藏状态的电网。重点阅读第3节和第5.1节。

### SPACY 把潜变量与关系作为发现对象

Wang、Varambally、Watson-Parris、Ma、Yu，*Discovering Latent Causal Graphs from Spatiotemporal Data*，ICML 2025主会。[正式记录](https://proceedings.mlr.press/v267/wang25cy.html)。

针对时空网格观测，联合学习空间聚合产生的潜变量及其关系图。可辨识性依赖空间因子、潜序列和可逆非线性观测映射等假设；证据包括合成结构恢复和气候中的已知现象。它为“状态也可成为发现对象”提供例子，但不是电网隐藏状态的现成解法。此处对其范围的概述依据正式摘要，未将其条件性结论外推到本项目。

## AI4Science 的问题定义与评价

AI4Science包含提高科学计算效率、学习有效预测模型、发现方程或机制、寻找数学证书，以及主动选择实验等不同任务。具体贡献应按科学产物定义，不能由网络结构、自动运行或“AI Scientist”称呼决定。

Vafa、Chang、Rambachan、Mullainathan的ICML 2025论文 *What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models*，检验预训练模型向由假定世界模型生成的新任务迁移时的归纳偏好。其案例表明，训练任务表现好与获得底层机制可以分离。对项目的启发是把预测能力与机制证据分别设计。[正式记录](https://proceedings.mlr.press/v267/vafa25a.html)。

Chen等的 *PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors*，发表于NeurIPS 2025 Datasets and Benchmarks轨道。智能体在有限实验预算内选择可控变量、提出方程，并在不同先验暴露程度下接受评价。基准分别考察方程等价与拟合，揭示“知道系统名称后调用已有公式”与“通过实验区分规律”的不同要求。其等价判定含符号工具及LLM辅助，解释成绩时应保留该评价条件。[正式记录](https://proceedings.nips.cc/paper_files/paper/2025/hash/42abcf9dffa48ca44fea1c497539a914-Abstract-Datasets_and_Benchmarks_Track.html)；[全文](https://arxiv.org/html/2507.15550v1)。

Zou、Feng、Zhao、Shi的 *Analytical Lyapunov Function Discovery: An RL-based Generative Approach*，发表于ICML 2025。动力学已知，学习对象是解析Lyapunov函数；生成器结合反例反馈，最终用形式工具验证。它与项目L2相关，但发现证明函数不等于发现运动方程。原文包含指定参数的两节点电网局部例子，不能据此推断IEEE39或40维模型已能得到同类证书。[正式记录](https://proceedings.mlr.press/v267/zou25a.html)；[全文](https://arxiv.org/html/2502.02014v3)。

NeurIPS 2025观点文章 *Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition* 将基础模型角色分成流程增强、人机共同提出问题及自主发现三个阶段。这是一种研究立场，不是已证明的能力等级。对本项目，自动执行实验的程度与实际发现的结构深度应分开评价。[正式记录](https://papers.neurips.cc/paper_files/paper/2025/hash/50896572558e01d1c6ad28018210554c-Abstract-Position_Paper_Track.html)。

## 最新相关补充与出版身份

Yang等的 *Discovering Symbolic Differential Equations with Symmetry Invariants* 以预先给定的对称群构造微分不变量，并以其为表达式搜索的基本材料。它发现满足指定对称性的方程形式，并不发现该对称群。该文正式身份为TMLR 2026年3月；虽能在NeurIPS下载目录检索到标题，本报告不据此标为NeurIPS主会论文。[正式全文入口](https://openreview.net/pdf?id=9t1dEyYfPc)；[作者出版列表](https://jiankeyang.github.io/publications/)；[作者全文](https://arxiv.org/html/2505.12083v2)。

它的前序工作 *Symmetry-Informed Governing Equation Discovery* 则是NeurIPS 2024主会论文，以对称性约束压缩方程搜索空间。[正式记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/77fa0e7d45c6687f1958de0b31e9fc05-Abstract-Conference.html)。

## 项目实际拥有的发现能力

| 项目对象 | 当前事实 | 对发现主张的含义 |
|---|---|---|
| 状态与机器组织 | 每机四状态、十机40维和逐机记忆组织由方案给定 | 学会q初态不等于发现所需状态数或其唯一物理含义 |
| 图与作用范围 | 物理图、邻域计算和基本结构固定 | 当前没有自动发现新的连接关系 |
| 结构候选 | 0098预写八个模板组，按风险梯度逐步开启 | 属于有限机制库内的数据驱动选择 |
| 结构精简 | 0098实际运行563–570步，未达到800步剪枝节点 | 不能把最终7/8组称为经过完整剪枝的最简结构 |
| S23与F23 | 同一固定23项展开库，S保留参数绑定、F放松绑定 | 是参数化/约束对照，不是开放表达式搜索 |
| 0102–0106 | 主要优化训练策略、系数和初始化；0103人工指定开放一个耗散组 | 当前主线是结构约束辨识及具体结构假设对照 |

代码与记录：[逐机模板及模型](../experiments/local_memory_rom_0098.py)、[0098增长与剪枝实现](../experiments/train_local_memory_0098.py)、[0098结论](0098_local_memory_capability.md)、[S/F展开](../experiments/shared_closure_rom_0101.py)、[0102协议](task2_prediction_optimization_0102.md)、[有效路线](../docs/TASK2_after_loop_101.md)。

固定反对称J、非负耗散R或角度平移一致性，是本项目采用的物理先验。它们可以支撑发现，但不能作为“算法从数据发现这些性质”的证据。此前预测优化有助于建立可信的内层拟合器；它没有自动完成外层结构问题。

## 可用于下一轮路线讨论的问题

建议将科学问题具体化为：在有限因果观测下，哪些低阶闭合机制对于自主暂态预测是必需的，哪些机制能跨工况复用，以及数据能把它们区分到什么程度？这里的产物首先是可检验的有效降阶规律；其与原高阶系统真实器件机制的对应需要另行证据。

最贴近已有基础的方向是“受物理约束的闭合方程发现”：保留经核对的运动学、单位和图信息，将闭合项的存在、函数形式或一种明确的机制分解设为未知。允许的表达式或机制族仍需有限，但最终结构不应在训练前已经写死。每个候选要得到可比的系数与初始化联合适配机会。

另一方向是“跨工况共享机制发现”：在有实际机制差异且数据条件足够时，区分共享动力学与工况作用。应先核实哪些工况改变方程、哪些只改变初态，不能直接把现有故障标签当作DIF意义上的环境。

建议的发现验证包括：

1. 在有已知结构的受控任务上，检验机制项、关系或函数形式是否恢复，而不只报告预测误差。
2. 在真实高阶数据上检验候选结构能否跨独立工况复用，并与具有相近拟合误差的替代结构比较。
3. 检查删去某机制并重新适配后是否仍能解释数据；删除后不适配的变化只说明当前实现依赖该项。
4. 对结构不确定性给出重复运行或数据扰动下的选择稳定性；固定有物理意义的坐标后比较，避免把潜坐标的等价变换当成不同机制。
5. 若多个候选仍不可区分，报告可辨识的等价类别或设计能区分它们的实验，不强行宣布唯一结构。

明确发现对象后，再设计证据和有限搜索算法。L1预测、结构发现证据、L2稳定性证明分别记录。当前讨论不启动训练，也不增加父任务资源额度。

## 检索和追溯信息

通过用户指定的Sider Scholar进行候选发现，随后用PMLR、NeurIPS正式论文、作者全文和出版列表核实。主要检索词为physics structure discovery、dynamical systems symbolic discovery、conservation laws discovery、invariant function learning、symmetry groups flow matching和symbolic differential equations symmetry；限定近期记录后按与动力系统结构发现的相关性筛选。未以引用数排行，也未声称穷尽会议全部论文。

Sider检索记录：invariant function learning `e7ba20b3-4125-4681-a581-9869bdea16c6`；symmetry groups flow matching `0a899183-dc22-4aa4-86cb-0842d0a612ac`；symbolic differential equations symmetry `09265c6c-a56f-493d-80eb-6c7530d316f1`。文献标题按题名去重，会议身份以正式论文集为准。本报告没有产生新实验结果。

