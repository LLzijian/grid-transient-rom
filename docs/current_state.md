# 当前研究状态

任务状态：**RUNNING：2026-10-19按用户授权推进闭合机制改进**。原test关闭，TASK4暂停，所有模型仍为研究候选，默认模型不变。

[执行协议](../reports/closure_mechanism_20261019_protocol.md)、[本轮进展](../reports/closure_mechanism_20261019_progress.md)、[原方案](../reports/closure_improvement_plan_20261019.md)。P0已核对366条train与0102 S-O来源，完成完整GPU更新测时、40项相关接口/恢复检查、真实短FOM双步长探针。完整搜索费用与提案排名规则仍需核算，未宣称全计划可行性门槛全部通过。

**P1已知支撑对照已完成18/18，各512更新，累计6085.6472446进程秒。**登记留出归一化MSE≤0.01的门槛通过0/18；不进入未知结构搜索、不采用这些模型。原主作业已退出。Astra6（gpt-6-astra）已完成本轮前置分析，root采纳一次9项真实q初态诊断；见reports/astra6_round_20261019.md。

工作目录E:/03-nn_Transfer_Func/kiki。批命令：`.venv-gpu/Scripts/python.exe experiments/closure_known_controls_20261019.py --output private_data/closure_mechanism_20261019/P1/known_controls --max-seconds 26000`。作业自动执行已登记的配置，在累计26000秒硬限前退出。若进程实际已退出，先读batch.json及日志，不能照此快照声称仍运行。本会话已启用30分钟心跳kiki-astra6；唤醒先检查真实进程，避免重复执行。

检查点在各配置attempt_*/known_support/checkpoint_XXXXX下，含模型/Adam/RNG；当前恢复方式是保留原尝试后显式整次重启，不支持逐更新续训。相同批命令复用累计时间、跳过已完成配置，源码身份不符则拒绝。诊断原PID52876因结果JSON写入故障已核对身份后暂停；不是模型失败。已记执行1444.3631872秒，保留node5101的512次与node5102的32次完整检查点；同状态恢复正在修复验证，尚未重启。原命令：`.venv-gpu/Scripts/python.exe experiments/closure_oracle_q0_20261019.py --output private_data/closure_mechanism_20261019/P1/oracle_q0 --max-seconds 4500`。状态/结果以该目录state.json为准；9项固定512更新，只提供100ms起点真实q，属于额外信息诊断，不可部署。当前新轮执行上限4500秒、检查汇总500秒，计入P1。下一动作是核验该诊断结果后重新调用Astra6规划，未知结构搜索及P2/P3/P4均未启动。

新总上限259200进程秒（72小时），P0/P1各14400/28800秒，已知支撑批作业累计上限26000秒计入P1。实际回执与保守支持预占见[费用快照](../results/closure_mechanism_20261019/cost_snapshot.json)，已知支撑批费用6085.6472446秒、诊断QA32.9762025秒；新诊断实时费用在oracle_q0/state.json/receipt.json；支持预占不是准确实际消耗，核对前不释放。旧A—E约17.403小时及旧TASK2的59643.306023/60000秒分别关闭，没有转移余额。

真实短历史探针两步长均取得11点100ms记录，预测起点实测为切除后100.1ms，标签unknown；归一化双步长差2.37654e-5。首轮终点舍入失败与修复重跑均留存。原30个绑定文件与index哈希保持。上轮A—E的[科学结论](../reports/closure_discovery_20261017_final.md)不变：生成式优势与部分观测机制恢复尚未成立，完整24条件确认未决，未获得非线性稳定域证书。

GitHub归档分支codex/closure-improvement-20261019；新源代码、协议、正负探针及紧凑证据归档，原始数组/中间检查点留本地。GPU工程与旧研究已合并main的9fe0ff2，本地HEAD a951013及用户未提交工作保持。当前为已完成控制的阶段审查与新轮次准备；每轮先调用Astra6，合格进展通过PR合并main，失败方案撤回采用并保留证据。
