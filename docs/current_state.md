# 当前研究状态

**2026-10-31已连续完成两轮实验与独立判读，自动接续已启用。**
用户要求按harness开展实验并自动迭代，既定方向有界科学推进已恢复；本聊天每小时接续kiki已启用。
新Run实际Astra设计、冻结有限协议、sol实现执行、独立审查、真实回执与远端归档逐轮保留。
原val/test关闭、新FOM=0、TASK4暂停；100 ms部署合同/default none/模型与工作指针不变。
本次两轮均无训练或新ROM预测；首轮10次离线共享参数估计，第二轮0拟合，不混称无拟合。

**最新结果。** 反推精度probe定位了父Run失败门的当前可复现来源：
同一binary64输入下，Torch64逐位复现旧p_backward，相对舍入高精度B_ref差1.56730184e−10。
SciPy64对B_ref差3.46945e−18，对未舍入MP120差1.06285e−16；两种误差口径已分列。
JSON读取相对已存binary64逐位一致，装配对照差8.67362e−19；历史唯一生成原因仍unknown。

**实际影响。** 两模型两步长均实际命中10次反推buffer调用，确为活跃路径。
4个对照格均297/297，8组真实初始化+8组初始场；递推/QR/场差恒等式均通过。
maxΔp历史4.60337e−9，maxΔr₀约1.30593e−9（模型坐标），maxΔωdot约8.61130e−9 rad/s²；
设计矩阵与末点观测逐位不变，Δṗ=0。无未来积分，不能由小初值差宣称长期风险或漏判不变。

**父Run结论保留。** p_port_diagnostic_20261031来源/平衡297/297对齐、基准差0，
M1自身闭合通过，M2六格为固定p残差与ω误差传导混合；原反推1e−12门未过，M3六十格归因unknown。
新probe不改旧门、不追认旧M3。上游tm标机械转矩，p仅按冻结数值坐标解释，不独立认证物理功率/能量。

**三决定。** 代码retain_branch；模型retain_evidence，副本丢弃、不晋升、不改默认或保护档案；
科学continue：数值来源/活跃路径/初值传递已在登记范围解释，p物理形式与长期性能仍未决。

**执行成本。** 首轮一次包装失败0科学工作+唯一技术修订后完成一套：原生wall累计116.7714384 s。
第二轮唯一原生启动exit0，wall6.7080261 s，工具总5.9114941 s、发布前5.7339353 s分别记录不相加。
第二轮17项人工测试；两轮当前测试共23+17通过。第二轮实际修复0，执行前代码审查修订单列。
CPU总进程秒、QA/LLM/全项目总成本unknown；原checkpoint/失败记录及关闭余额不变。

**材料。** [精度probe结论](../reports/p_reverse_precision_20261031_conclusion.md)、
[协议](../reports/p_reverse_precision_20261031_protocol.md)、[审查](../reports/p_reverse_precision_20261031_review.md)、
[公开结果](../results/p_reverse_precision_20261031/public_probe.json)；
[父p诊断结论](../reports/p_port_diagnostic_20261031_conclusion.md)已归档远端f48987a。
私有历史、残差、调用和原生身份留各Run私有目录，未上传。
各Run远端归档核验见其私有archival_verified.json，不由COMPLETED推断推送成功。

**下一动作。** 解析p_backward数值候选资格草案已由Astra准备，DRAFT_UNADOPTED；
最多固定A96/A16两个仅buffer改变的数值候选，原因果/主细/逐事件/T资格和原科学准入门分开。
采用新有限协议与独立审查后才可运行新paired预测；不由当前probe自动采用解析buffer或扩大结构。
每小时接续从该草案继续，不重复已完成/身份不明Job；用户后续暂停优先。

**保护。** checkpoint九点原全门失败、step96的真实区间20/30与8 s 4/30权衡、R_N保护档案均保留。
旧memory_transport关闭1356余额、r_memory INTERRUPTED_RUNTIME及UNKNOWN全项目账目不改。
