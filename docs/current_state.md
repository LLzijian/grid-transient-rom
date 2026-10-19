# 当前研究状态

任务状态：**RUNNING：闭合机制研究持续推进**。工作目录E:/03-nn_Transfer_Func/kiki。原test关闭、TASK4暂停、默认模型不变。所有新增模型仍是研究候选。

**节点因果初始化比较已CLOSED，原PID45576退出，无训练作业或重复队列。**三个种子×原GRU/历史残差共6项，各512更新，全部完成且训练后双步长合格。合成val MSE：对照0.163873/0.220874/0.236702，候选0.00148505/0.00228073/0.00234259；候选3/3达到原0.01门槛。这里为16train、16合成val，0.5/1s未来；结构已给定，不是未知公式恢复或真实电网部署。见[结论](../reports/node_causal_20261019_conclusion.md)、[证据](../results/closure_mechanism_20261019/node_causal_summary.json)。独立只读挑战node_result_challenge已接受限定结论，root采纳为研究证据。

本阶段[协议](../reports/astra6_node_execution_20261019.md)的15600秒执行限内实际记7475.150711秒；原目录private_data/closure_mechanism_20261019/P1/history_residual/node_run01禁止重跑。入口experiments/closure_causal_history_residual_20261019.py，源SHA256 0b217d44fde452f31e517625d33c4522e658591ae9d7c5072e8bdf42a489744a。各组checkpoint_00512包含模型、Adam/RNG，逐事件结果/数据/来源哈希在紧凑证据中；原始轨迹和训练文件留本地。只接受完整核账的捕获中断恢复，未知强杀/孤立检查点拒绝自动恢复。

P1此前18个原GRU已知支撑对照全部完成512，共6085.6472446秒，预测门槛0/18，不重跑；真实q诊断在4500.0271288秒原限关闭，节点3/3、边2项通过，边1项未完成、饱和未启动，不追加诊断。历史残差费用预检未通过原10800执行档，未训练且取消其未启动分配，1200秒检查分类保持扣留。

预算：父259200秒、P0/P1上限14400/28800不变。P1保守累计20560.8250844秒（控制6085.6472446+诊断4500.0271288+其检查500+费用预检检查1200+节点执行7475.150711+节点检查800）；剩8239.1749156秒。节点未消耗的8124.849289秒执行分配明确撤销，仅当前新研究未用额度可前瞻重新分配。P0已测142.1922393和旧900秒未核对支持预占保留，父余额至少237596.9826763秒。检查预占不称准确消耗，旧关闭预算不转移。见[费用快照](../results/closure_mechanism_20261019/cost_snapshot.json)。

**下一轮已实际调用gpt-6-astra并采纳edge六项配对方案**，见reports/astra6_edge_execution_20261019.md；node/edge混用需禁止。固定edge三种子×两组×512，执行6600+检查800=7400秒，从P1余额8239.1749156分配，分配后保守839.1749156秒；父保守230196.9826763秒。固定edge6_6600顺序/预算/恢复隔离检查9项通过，并已独立复核、归档。实际已启动edge_run01，PID13468，RUNNING/fit/edge_5101_control；源SHA256 abe887a7c4758f0ad21fa0d9fb97282ce22349a03eb70a09bbb478a8f54f364a。命令：`.venv-gpu/Scripts/python.exe experiments/closure_causal_history_residual_20261019.py --output private_data/closure_mechanism_20261019/P1/history_residual/edge_run01 --profile edge6_6600`。状态/累计费用以该目录state.json/receipt.json和真实PID为准；下一动作稀疏核验，六项拟合冻结后评价。禁止新源码恢复已关闭node端点。节点结论不推广到边/饱和，未知结构搜索及P2/P3/P4关闭。30分钟心跳继续，健康作业保持安静。

P0核对366条合法train与0102 S-O来源，GPU接口/恢复及真实短FOM双步长探针完成：11点100ms、切除后100.1ms起点、标签unknown，归一化差2.37654e-5；不等于完整事件资格。完整搜索费用尚未锁定。旧A—E约17.403小时、旧TASK2 59643.306023/60000秒各自关闭。历史生成式优势、完整确认和非线性稳定域结论保持未决，见closure_discovery_20261017_final.md。

工程及协议已通过PR #5合并main 0cabe2e75240c141319a4bcea2aff25a21828e81；本轮结论将选择性归档。原本地HEAD a951013和Git index/用户未提交工作保持，不重置或强推。恢复入口为本页、当前协议与决定性证据；不需重读全部历史。