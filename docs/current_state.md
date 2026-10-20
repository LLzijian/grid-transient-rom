# 当前研究状态

任务状态 **RUNNING**；E:/03-nn_Transfer_Func/kiki。无在运行研究作业。两种边记忆结构比较已关闭，下一动作是实际调用Astra6制定机制删除重适配的有限方案，登记后执行。项目val/test关闭，TASK4暂停，原科学门槛未变，任务L1/L2尚未完成。

**刚完成**：[两结构结论](../reports/edge_memory_discrimination_20261021_conclusion.md)。旧正弦三个512端点复用，新余弦三组512完成；统一冻结后各一次16train/16合成val双步长评价均完成、退出0/GONE/源一致。六端点事件平均归一化MSE均≤0.01，train三个种子均选正弦；不代表每事件通过或唯一机制恢复。独立结果挑战已采纳。冻结SHA d1706d146857946a771596adbc3576df5fb2b0014fd6f21ceb8a7aa9064a1c61；实际代码b5b88fb8994c9adee4f552cff56586f03467c568，科学runner SHA93a579962ae6383df292913a0ff6a81d2a31a9c44d40e8dc5115860b214275f8。源检查点和完整恢复文件在private_data/closure_mechanism_20261019/P1/edge_memory_discrimination01，不重跑关闭作业。

**预算**：父259200上限，累计收费或保守预占136565.71861110005，余122634.28138889995秒。当前轮P0关闭930.5387226/P1关闭1362.7101726；剩余P0 7192.2323664、P1 2619.3802045、P2 3060.8945846、P3 75986.9220123、P4 21600、repair12174.8522211，不互借。支持预占不冒充精确实耗，旧900/P8尾/W尾40均保留。[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。本轮尚未启动任何新规划或计算预占。

**保留范围**：两个边结构是合成研究候选，不是185项未知结构恢复、真实模型或盲确认。未证明新增边项必要，需要整个机制删除后重适配。三余弦512仍在改善，不称模型族最优。默认模型不变；本轮专用试验归档，main不集成候选。此前合格工程[PR23](https://github.com/LLzijian/grid-transient-rom/pull/23)已合main27aa39ad2a2821adefbee1913377a9af153c868c并验证，W8科学门槛FAIL保持，无需撤销工具。

**边界与恢复**：40状态/100ms一次GRU/真实区间/单位保持；真实train385=342稳定43失稳。原18×512的6085.6472446、q0/node/edge、饱和及其他已关闭作业不复活。P1广搜门槛未过，未来有限受控研究可前瞻登记；P3全部配额用尽/旧24valguard不重试不搬模型；P4缺同候选L1。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb/index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93未改。heartbeat ACTIVE30min。
