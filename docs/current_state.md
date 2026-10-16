# 当前研究状态

任务状态：**RUNNING，阶段C公平重训**。实施用户批准的72进程小时物理闭合方程发现计划；[协议](../reports/closure_discovery_20261017_protocol.md)、[资源变更](../reports/closure_discovery_20261017_resource_amendment.md)。TASK4暂停，原冻结test关闭。

A已形成可执行首版，原S-O向量场重建误差2.22e-16；两个新增train条件通过双步长FOM核验。已知q组件例与100ms部分观测例的证据范围分别记录，后者不冒充正确公式恢复。详见[阶段A结论](../reports/closure_discovery_20261017_stage_A.md)。

B六次搜索已结束，11个合法512次端点、三个数值失败均保留。固定库与生成式均选同一八项模型，删除port es，掩码[1,1,1,0]；不能据此声称生成式优于固定库。10项替代候选满足近优及可观测分歧规则，其新增(p+q)p在当前B端点可吸收到原p二次储能和pq系数。独立只读审查同意继续原C比较，不晋升最终模型。详见[阶段B结论](../reports/closure_discovery_20261017_stage_B.md)。

当前绑定源码2e38331099ecb14bfe0d858e1cc88671fba996264bba88bd74ccfc36f017e487。repair03通过67项定向回归，不改训练数学或科学预算；只读B缓存汇总禁止fit且原模型/优化器/失败字节保持。旧f27实现及三种子固定对照保持原来源、1024次和SHA。

实际C进程：main2701 PID49072/工具会话27680；main2702 PID47548/29981；library2701/2702/2703顺序执行 PID14252/34842；alternative2701/2702/2703顺序执行 PID46048/65518。均按原1024更新及固定后半程衰减。详细命令、执行单与receipt见 `private_data/closure_discovery_20261017/C/active_refits_snapshot.json`；快照时间 2026-10-16T22:51:35.223797+00:00。C固定2701/2702/2703已完成且物理QA通过。

下一可执行动作：等待main2701完成；在释放的进程槽对其八个完整表达式分别删除、固定其余权重评分，只作提案排序；root按原风险及索引降序选择最多两个，登记并推送后从同一S-O各重训1024次（seed2701）。之后完成main2703。全C合格端点形成冻结清单后，D生成新增train24、统一追加256次、冻结全部权重后才生成新增val24。D/E科学执行尚未开始。

总授权259200进程秒；分配A43200/B57600/C86400/D43200/E14400/reserve14400。本页闭账支出 A=1986.0352545, B=15432.0706683, C=6777.0731551, D=26.6544076, E=41.0764439, reserve=202.1213282；四条在运行C进程尚未关闭，其实际时间退出后追加，不能将闭账余量当成全部可用预算。唯一账本 `private_data/closure_discovery_20261017/ledger/receipts.jsonl`；旧TASK2的59643.306023/60000秒单列，不转移额度。

恢复目录 `E:/03-nn_Transfer_Func/kiki`，Python `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe`。来源0102 S-O SHA256 c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238。S-O是研究候选、F-O是开发基线；train为346合格事件、含35个失稳样本，真实区间与100ms因果边界不变。

GitHub归档分支 `codex/closure-discovery-20261017`，最近已核实提交 d4bdd5f8f8eb15aa721150b4e54dd33c4cc360f4。关键设计、代码、正负结果均归档；原轨迹、缓存及中间检查点留本地，重要模型列路径与SHA。无关修改与本地HEAD a951013保持原样。工程接口见[执行说明](closure_discovery_runbook.md)；历史状态见[备份](history/current_state_before_closure_discovery_20261017.md)。
