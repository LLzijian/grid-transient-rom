# K4稀疏静态图：独立实现审查

当前结论：`revision_required_not_released`。发现一项有界合同偏差：跨网格数值资格未比较完整秩标记。其余数学、支撑资格、隔离与计数路径未发现阻断；先修复这一项再复核，不生成正式轨迹或改变科学门。

冻结输入：runner `experiments/gftd_sparse_topology_20261107.py` SHA256 `F5701306574EBAA15EA050550911CB99ABA182267479E7AC9C71753D5B5E2594`；测试 `tests/test_gftd_sparse_topology_20261107.py` SHA256 `A1B9A17EC0147A4E5CB26C8557BA15F47E3CD2E0449B4D6BB9EFD12FCB4EE24C`；协议 `040B03BEC2BF2021DF7BB4FF1D18BA769095D5023AA73E6178CE38924E510368`，最终设计审查 `1197D610FCB921BFC7948FA7F4C5977B062AED864B8656CDBF5AC60575E86544`。

## 必要最小修复

第230–254行qualify_pair中，第234–236行只比较两网格的rank_threshold_sensitive布尔值。两格可以各自三个阈值下秩稳定、布尔均false，但实际秩不同，仍被记numerically_qualified=true；这不符合协议的秩标记一致。完整支撑谓词各自要求六列满秩，故主要exact-support采用另受保护，但未决诊断仍需准确。

应比较完整rank signature：六列ranks、记忆投影memory_ranks及六个条件子字典other_ranks，或等价全部标记。保留两格均insufficient的负校准分支；不改阈值、NNLS或支撑规则。新增纯模拟反例：两格敏感布尔均false但稳定秩不同，必须数值未决。当前版本不释放。

## 已通过的静态核对

- **方程、坐标与scale。** K4边序和b_e=e_i-e_j正确；N=4的完整20维状态，r力端口系数为1、r_dot=.7omega-.3r，p_dot=-omega-.3p。weak_terms逐边列及b=惯性-D+p-r_f符号正确；Z按rho/.1缩放与力积分/.1抵消。记忆投影后X/y同时除sqrt总行数，不逐列改惩罚尺度。
- **A资格继承。** 主细弱残差、两格差、生成相对theta/完整omega及p末点门完整；p/r重建两格报告，仅细格决定1e-7资格。端口激励采用父合同解释：全非零条件池内p/r各至少一项绝对积分>1e-8；每条件/每格terms保留可复核，不新增每条件必须非零的门。root已采纳此释放前解释。
- **NNLS与失败传播。** 全64子集含空集，J、lambda和并列窗口与协议一致；代表按边少、字典序，不冒称多并列唯一。KKT使用实际子矩阵的数据尺度，精确零与空集单列。任一枚举或去偏未决传播整设置enumeration_unresolved，不能删除失败子集后宣称全局充分。raw代表可保留诊断，支撑资格仍被阻断。一次.05阈值及一次去偏，跌破阈值保留权重并标不一致，不循环筛选。
- **仅观测支撑资格。** 两格各检查Frobenius、包括未选边的全部六列绝对强度、三个阈值满秩、记忆及条件子字典秩敏感性、条件残余比、相关、全求解、算法唯一和去偏一致性。失败正式support=null；信息合格且选零边才=[]。algorithm_unique与support_qualified分开，评分器只有在资格成立后比较真支撑；null的FP/FN仍为null，raw误差另列。
- **数据隔离。** 白名单仅theta/omega11点、times、p_end、sigma；build_system使用固定非k参数。生成真k只进入人工生成、A及冻结后的评分；plan提供预定批次和组别，但infer_setting只接A/Z/b/sigma，不用E0标签控制返回。全部正式样例保持未运行。
- **工作量。** 32生成格、48包、96重建、48设置；3120代数项与最多3072实际NNLS区分。整体信息不足时65项标skipped且保留记录，最终核actual+skipped=3120；空集及空去偏无NNLS。attempt/completed/qualified计数分别保存。
- **闭环。** 协议及释放源哈希核对、root released门、独占started、A失败不执行B及持久receipt、estimates和观测/plan哈希冻结后才读真值评分、最终计数和source_unchanged均存在。提前停止A不是科学恢复成功；进程状态与科学判断分开。

project_memory保存投影正交性数组而不新增运行时阈值，与父实现的“保存并由独立结果矩阵复核判断充分性”一致。结果阶段仍须按保存A/Z/b和全部子集解独立审计相对KKT/投影误差、全64目标和并列、去偏及完整资格，不能把SciPy返回成功当作充分性证据。

root/Modeler报告19项纯代数/mock资格通过；Reviewer只读最终源码与测试，未重复运行。测试覆盖K4端口、非法字段、相对KKT反例、64枚举、空集/null、未选未激励边、共线并列、任一子集失败、跨阈值敏感、去偏阈值不一致、mock种子与48/96计数及冻结哈希。待补完整跨格rank signature反例。

代码决定：完成上述最小修复后再接受；模型不晋升；科学执行当前未释放，不预判四个无噪批次成功。Reviewer未调用main、生成正式16轨迹、消费150000族、读取私有事件或修改源码/测试。本审查不是新增独立实验证据。

## 最终修订接受意见（替代上述当前决定）

最终runner SHA256 `2EEBB1E659E1B17CACD5ED168745BF1638101195937BAD70CF708485F55F7C7B`；最终测试SHA256 `152CC4D22A25022BD6E9495AF9426603937592E184C036748A597D3B1FDE3EC5`。`results/gftd_sparse_topology_20261107/prelaunch_v1/`中的旧源/测试字节仍分别匹配原F5701306574EBAA15EA050550911CB99ABA182267479E7AC9C71753D5B5E2594、A1B9A17EC0147A4E5CB26C8557BA15F47E3CD2E0449B4D6BB9EFD12FCB4EE24C。

已逐行diff：唯一源码改动是在正常跨格数值分支增加完整rank signature比较，依次包含六列ranks、memory_ranks及六个条件子字典other_ranks，均按三个固定阈值比较。原both-insufficient的短路负校准分支保持不变；支撑、阈值、惩罚、数据与预算没有改动。

测试仅新增8个参数化模拟例，覆盖full、memory及每个条件子字典稳定秩不同、敏感布尔均false时必须数值未决。root/Modeler报告最终27项通过，pytest1.44 s、exit0；Reviewer只读测试，未重复执行。

必要修复已落实，无剩余阻断，状态更新为 `eligible_for_root_release`。允许root按最终源和既有协议清单释放唯一有限执行；模型不晋升，不预判支撑恢复成功。结果仍须独立复算A资格、全子集KKT/J、并列/去偏、支撑资格及实际/跳过计数。最终复核未运行科学、采样、测试或读取真实事件；报告至此停写。
