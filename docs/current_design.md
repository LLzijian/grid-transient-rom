# 当前架构 · OSGKD 2.0（已封卷参照）与 PDD v3.1（Scientist 已审定，待 AGY 复核）

> **2026-11-28 架构切换筹备**：基于 OSGKD 2.0 封卷证据与定位审查，项目已正式立项面向 NeurIPS 2027 的新主线 **[PDD v3.1（OSGKD-Decoupled）架构简报](../designs/pdd_v3_architecture_brief.md)** 与 [PDD v3.1 Campaign](../campaigns/pdd_v3/campaign.md)，将头条问题切换为“训练期特权物理观测（$\mathcal{O}_1$）与部署期短时部分观测（$\mathcal{O}_0$）下的开放网络化动力学可辨识性与两阶段解耦发现”。v3.1 已获 Scientist（`gpt-6-astra`）审定与静态代数核查，AGY 独立复核及用户最终放行仍待完成；在放行前，本页仍保留已封卷的 OSGKD 2.0 端到端规格作为历史与消融参照，`default none` 不变。

从开放异质惯性振子网络的短时运动观测中，哪些耦合、非线性力和记忆阶次可以被区分？OSGKD 2.0 用同一显式方程解释历史、预测未来并计算开放功率收支。IEEE39 的部署合同仍为 11 点、100 ms 相角和完整转速，加冻结常数。真实机械初值只用于另列的信息增强对照。[算法规格](../designs/osgkd_v2_architecture.md) · [信息合同](../kiki.md#motion_only_v1)

| 设计选择 | 可检验的问题 |
|---|---|
| 内部互易势、独立的驱动与环境端口 | 允许环境净交换后，哪些边、调制和高阶谐波仍有作用？ |
| 节点共享的初态器及历史窗物理精修（EAVI） | 与同一动力学的纯编码、纯迭代相比，部署时估计和可观测未来能否同时改善？ |
| 先拟合完整字典，再删组重拟合 | 给较简单模型同等适配机会后，哪些结构仍有必要，哪些只能识别到等价类？ |
| 长短轨迹联合训练 | 从短时域到长时域学习时，`val` 稳定 8 s 风险与 8 s 自主失稳召回能否同时改善？ |

OSGKD v1 的封顶基线是原方程第 2260 次，50 维状态和 6,174 参数的扁平 GRU。高阶势与 60 维零和快记忆作为扁平框架下的探针封存。Sprint 1 保留原开放动力学的三部分结构，以 1,928 参数的节点共享图初始化器从头联合训练。图初始化器通过对称性和同权跨规模门禁，两点 `val` 自主召回为 9/12，稳定风险超过预登记门。代码和两端点作为研究证据归档。[图初始化器设计](../designs/osgkd_v2_p2a_equivariant_graph_20261124.md) · [图初始化器结果](../reports/osgkd_v2_p2a_equivariant_graph_20261124/result.md) · [封顶结果](../reports/osgkd_v2_sprint0_20261124/result.md) · [运行状态](current_state.md)

OSGKD 2.0 的默认配方为完整 `train`、每批 1/8 失稳、0.4/0.3/0.3 分段运动损失、批内事件等权均值和余弦退火。主评价用 8 s 自主失稳召回，与稳定 8 s 运动风险、误报数和原始分母一起读取。真实区间检出和触发时间作为诊断。[评价修订](../campaigns/osgkd_v2/evaluation_amendment_20261124.md)

Sprint 2A 比较三种子图初始化器与同种子扁平 GRU。原 C1 方程、宽幅值界和训练条件共用。依据 Sprint 1 第 2048 步图模型的正实增长模态，四组在 8 s 阶段约束固定参考点的闭环谱。原始 `val` 上，图 `24121` 的稳定 8 s 风险为 0.4309、误报 0/95、自主召回 9/12。三个图种子的平均稳定风险和召回均未达到晋升门槛，暂留四组证据。同种子扁平对照保留为本轮比较锚点。[匹配设计](../designs/osgkd_v2_sprint2a_20261125.md) · [前瞻协议](../campaigns/osgkd_v2/sprint2a_spectral_matched_protocol.md) · [结案报告](../reports/osgkd_v2_sprint2a_20261125/result.md)

Sprint 2B 的删组和同条件基线检验互易交换、记忆和开闭端口对预测的作用。Layer 1 使用独立已知真值的开放振子网络度量边和调制恢复，两机反例单独检验有界核与无界核的有限时行为。[备战架构](../designs/osgkd_v2_discovery_preparation_20261125.md) · [稀疏与基线协议](../campaigns/osgkd_v2/sprint2b_discovery_baselines_protocol.md) · [合成协议](../campaigns/osgkd_v2/sprint3_layer1_protocol.md)

当前规范与修订证据：[v3.1 Scientist 规范](../reports/pdd_v3_architecture_20261128/v3_1_scientist_spec.md) · [修订台账](../reports/pdd_v3_architecture_20261128/revision_ledger_v3_1.md)。旧 v3.0 审阅结论只对应其冻结输入；当前执行仍暂停。
