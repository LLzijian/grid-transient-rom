# 当前研究状态

任务状态：**RUNNING：按用户授权推进闭合机制改进**。原test关闭，TASK4暂停，默认模型不变，所有新增模型仍为研究候选。工作目录E:/03-nn_Transfer_Func/kiki。

[执行协议](../reports/closure_mechanism_20261019_protocol.md)、[Astra6计划与采纳](../reports/astra6_round_20261019.md)、[恢复证据](../reports/oracle_recovery_20261019.md)。P0核对366条train与0102 S-O来源，完成完整GPU更新测时、接口/恢复检查及真实短FOM双步长探针；完整搜索费用和提案规则尚未锁定。

P1已知支撑对照18/18均完成512更新，累计6085.6472446进程秒；留出归一化MSE≤0.01门槛通过0/18，不采用这些模型，不进入未知结构搜索。原主作业已退出，无需重跑。已调用gpt-6-astra登记一次9项真实q初态诊断，各512更新；只在100ms起点提供真实q，属于额外信息诊断，不能部署。

**真实q诊断已按原执行上限关闭，PID80888已退出，无训练作业在排队。**9项中5项完成512更新且通过预测/数值门槛（节点三个种子、边两个种子）；边5103保存160更新，未评分；饱和三项未开始，不延期补齐。节点支持下一轮集中因果初始化，额外真实q信息不能作为部署成绩。见reports/oracle_q0_20261019_conclusion.md及results/closure_mechanism_20261019/oracle_q0_summary.json。

关闭诊断复现入口为experiments/closure_oracle_q0_20261019.py，原闭合目录禁止续训。执行累计4500.0271288秒，含两次中断与最后1911.0425666秒，0.0271288秒截止退出开销照实计入；不再加上receipt的prior。检查/汇总分类保守扣留完整500秒，旧900秒支持预占保留。

父上限259200进程秒，P0/P1上限14400/28800秒；新父预算下P0已测142.1922393秒、原900秒支持预占、P1对照6085.6472446秒及本诊断累计记账，见[费用快照](../results/closure_mechanism_20261019/cost_snapshot.json)。旧A—E约17.403小时和旧TASK2的59643.306023/60000秒各自关闭，没有转移余额。receipt的cumulative_seconds已包含两次中断费用，禁止再次相加。

历史残差原费用预检关闭：9次GPU完整更新最大3.7477823秒，原10800秒执行分类不足，未启动正式拟合；1200秒检查分类保守扣留，未启动10800秒分配明确撤销，不改旧门槛。Astra6已重新实际分析，同一研究问题进入节点执行阶段，见reports/astra6_node_execution_20261019.md：node三种子×两组各512更新，共6项；全新profile node6_15600，15600执行+800检查=16400秒，P1上限28800及父259200不变。分配后P1保守114.3256266秒、父229472.1333873秒，旧900秒支持仍保留。模块及新增预算profile的8项检查通过，正式作业已启动，实际PID45576，RUNNING/fit，当前首项node_5101_control；工作目录E:/03-nn_Transfer_Func/kiki，源SHA256为0b217d44fde452f31e517625d33c4522e658591ae9d7c5072e8bdf42a489744a。命令：`.venv-gpu/Scripts/python.exe experiments/closure_causal_history_residual_20261019.py --output private_data/closure_mechanism_20261019/P1/history_residual/node_run01 --profile node6_15600`。实际状态/累计费用以该目录state.json/receipt.json和真实PID为准，下一动作稀疏监测；六项冻结后才评价，不按val改配置。仅已捕获完整核账中断可恢复一次，未知强杀/孤立检查点不自动重启，保留并核账。原test/未知结构搜索关闭、TASK4暂停；禁止重复队列。

真实短历史双步长探针取得11点100ms，预测起点为切除后100.1ms、标签unknown，归一化差2.37654e-5；不等于完整事件资格。原30个绑定文件及Git index未改。旧[科学结论](../reports/closure_discovery_20261017_final.md)保留：生成式优势、部分观测机制恢复及完整确认仍未成立，尚无非线性稳定域证书。

格式修复PR #4已合并main a1e3b4c，模型不采用；关闭证据及Astra6新协议已归档codex/causal-history-20261019的ea7cf23/e3df0b8；实现检查通过后通过PR合入。原始数组和检查点留本地，重要哈希归档。原本地HEAD a951013及用户未提交工作保持，不重置或强推。