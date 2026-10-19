# 当前研究状态

任务状态：**RUNNING**，目录E:/03-nn_Transfer_Func/kiki。当前quadratic_memory_edge固定函数研究已登记并实现：同源full8 global1280构造A原项/B删除/C二次记忆项，三seed九组、统一新Adam，保留100ms一次GRU及1/2/5s循环。[协议](../reports/quadratic_memory_edge_20261020_protocol.md)、[实际gpt-6-astra原案](../reports/astra6_after_fresh_val_failure_20261020.md)、[独立审查](../reports/quadratic_memory_edge_20261020_implementation_review.md)。QA01 26通过/1失败，唯一guard修复后QA02 27项全部通过；尚未正式训练。当前无科学PID，下一实际执行为prepare源/RNG/共同385日程、随后cost实际时测，通过才启动全部九组512（可选全九768）。

**费用**：父上限259200；已关闭70318.2526126，本轮支持hold3000+两QA原生23.6409882，当前累计73341.8936008、剩余185858.1063992；本轮P2预占36000，未用32976.3590118，父未分配152881.7473874/P2未分配30100.6537796。工程2500/训练27000/评价3500/支持3000不可互借；单模型全部尝试≤3000，外部恢复1200嵌套一次，唯一实现修复已用。旧900未核对支持hold保留，子进程不双计。[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**边界与既有结果**：fresh_design_pair已关闭并归档8aa736ce2104079d6931d11244248c69beb5f31d；共同385=原366+新19（342稳定/43失稳），新5失败保留。六模型1280、18次train评价有效范围见[关闭结论](../reports/fresh_design_pair_20261020_conclusion.md)。删除组稳定train风险与失稳误差存在取舍，6101/2 delete7原事件expansion_0094的8s未决不改写。val路径接口在求解前拒绝，0物理条件/0轨迹/0成绩；旧24不重试、不搬模型、不放宽旧guard，本轮无FOM或val。原test关闭、TASK4暂停、生成结构搜索关闭、默认模型不变。

**恢复与归档**：新BASE=private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01；当前代码来源ee922ca3e0640768a8d1dea5d4e3f8bb4382a1a6，QA02实际native session98567已exit0且worker清理GONE，270源前后一致。精确命令、绑定、回执见BASE，公开分支codex/quadratic-memory-edge-20261020；私有helper归档副本需还原至该BASE，native测试不声称公开clone直接自足。失败QA和源模型不删。main保留6b17be3535cc57985c1e1be4138decabc7c0f033，尚无本轮PR；工程合并与模型采用分开。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb/index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93及用户工作保持。原18组GRU512已完成6085.6472446不重跑，heartbeat ACTIVE，健康运行无变化保持安静。
