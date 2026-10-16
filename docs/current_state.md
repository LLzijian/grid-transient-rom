# 当前研究状态

任务状态：**RUNNING，阶段B完整结构搜索**（2026-10-17）。执行用户批准的72进程小时物理闭合方程发现计划；协议为[执行协议](../reports/closure_discovery_20261017_protocol.md)，配置为 `configs/closure_discovery_20261017.json`。TASK4保持暂停，原冻结test关闭。

已接通表达式生成、物理编译、联合拟合、受控问题、新FOM数据与冻结确认、NumPy独立导出。59项针对性检查通过。原S-O向量场重建最大差2.22e-16；两条新增train条件通过双步长核验，物理系统指纹与历史train一致。受控问题证据和范围见[阶段A结论](../reports/closure_discovery_20261017_stage_A.md)；B已开始，尚无搜索终点、正式重训或新val成绩。

实际进程：**B正在运行，PID 46092**，工具会话30607，命令 `experiments/discover_closure.py B`。启动与代码身份见 `private_data/closure_discovery_20261017/B/job_receipt.json`，实时状态见 `active_job.json`。固定一轮、两方法各三个种子；候选32→128→512更新。下一动作是每3–5分钟检查存活并在128步等科学节点读取进度，B结束后作阶段决定并归档，再执行C。C/D/E尚未启动。

新授权总上限259200进程秒；前瞻分配为A43200/B57600/C86400/D43200/E14400/reserve14400，详见[资源变更](../reports/closure_discovery_20261017_resource_amendment.md)。阶段A已结算1986.0352545秒；B启动及运行另外累计，结束时统一结算。总授权在扣除A后为257213.9647455秒，B阶段上限57600秒。唯一费用账为 `private_data/closure_discovery_20261017/ledger/receipts.jsonl`。旧TASK2的59643.306023/60000秒独立保留，不转移、不重置。

固定来源为0102 S-O：`private_data/task2_prediction_optimization_0102/formal_S_O/step00650/structured_numeric_export.json`，SHA256 `c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238`。S-O是结构化研究候选，F-O是普通稀疏开发基线；均未成为最终部署模型。train为346个合格事件、其中35个失稳样本；历史val为107个合格事件、其中12个失稳样本。保留原标签、真实时间和100ms输入边界。

恢复目录 `E:/03-nn_Transfer_Func/kiki`；Python为 `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe`。入口命令按阶段为 `experiments/discover_closure.py B`（之后C/D/E）。B必须先通过最终成本锁定；断点包含模型、优化器、随机状态、采样游标和实际更新数。部署与能量/稳定性声明尚未形成。

GitHub归档分支为 `codex/closure-discovery-20261017`，代码与阶段A归档已核实提交 `8f1958749afd7a50c41cd1faa44ad2f1904e065b`。归档包含59项检查、关键正负结果及字节级依赖快照；不要重开已完成核验。关键设计、代码及正负结果均须推送并核实；原轨迹、缓存和重复检查点留在本地。无关工作区修改保持原样。

历史结论与余额见[前状态快照](../reports/history/current_state_before_closure_discovery_20261017.md)及0102–0106结论。历史accepted/failed分类不变；旧父预算耗尽状态不适用于本轮新增授权。
