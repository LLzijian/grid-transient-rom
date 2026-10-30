# 当前研究状态

**PAUSED_BY_USER。** 用户要求本轮完成后暂停；2026-10-30继续授权的工作为实验材料整理、GitHub同步与Git清理，不启动新的科学Run。原val/test关闭，TASK4暂停，新FOM条件0，默认模型none；L1高精度自治预测与同模型L2尚未完成。

**当前Run：memory_transport_20261029，COMPLETED。** A/S/P各128步，共492次optimizer尝试且全部完成（QA108＋正式384），repair/replay0、pending为空。共同128步决定NOT_RETAINED，无延长、无第二种子；未用1356已关闭且不得转入下一Run。三组真实区间≤15s漏检均21/30、8s均4/30、误报均0/267；范围为train fit297（267稳定、30失稳）。详细指标、分母、曲线、成本和三个决定见[本轮结论](../reports/memory_transport_20261029_conclusion.md)与[最终公开包](../results/memory_transport_20261029/completed_stage_public.json)。

**运行已收尾。** prepare/qualify/qa/study/summary五阶段均exit0，清理核验通过、Job内活动进程0、源码前后相同；活动锁已移除。临时保活任务kiki-memory-transport已删除，配置无残留，无运行中的本轮科学作业。成功五阶段合计wall7472.720580s、Job CPU12326.25s；不含外层controller、首次启动失败、metadata冻结及前置数学/守卫/平面准备，未知成本不记0。首次启动失败与全部恢复证据原样保全。

**来源与指针。** 执行源`181d7f8bba2eb530a0ad5afe0130035755e974a2`；逻辑位置`memory_transport_20261029/study01_prelaunch_retry1`，来源N_M192→SGM A_28311 step96（A96），参照R_N。binding SHA256为`edc9863941d2febae32bb799b85fa9969c84e50f511cdb13ed3ce57fd95768f8`，181文件，100公开源已核对。最终公开包SHA256为`f30159152865b672bcf3e61a031cbc491f2f5525a212f38bfda90d383192284f`。原始预测、checkpoint、Adam/RNG和native回执保持原位置，恢复命令保留在[协议](../reports/memory_transport_20261029_protocol.md)，暂停期间不得重启。

**档案与历史。** 可用模型档案及原接受范围受保护；R_N、A96、M-A512、交换候选及A/S/P端点的证据均保全，本轮没有晋升模型或自动重置研究起点。近期五阶段的问题、结论和材料见[近期实验导航](recent_experiments_20261029.md)。旧r_memory仍为INTERRUPTED_RUNTIME，1165实际/1152持久，原因和正式总成本UNKNOWN；不得用近期已知合计代替全项目账目。

**代码与科学决定分开。** 规则PR34、条件证明PR35已合入main；条件内域证明不是实际端点认证、完整L2或FOM安全保证。当前研究代码保留在研究分支，本轮不改变main科学模型。[Astra下一结构条件准备](../reports/astra6_next_structure_preparation_20261029.md)未采用、未注册、未分配预算。下一科学动作等待用户明确恢复；当前仅完成材料和Git归档收尾。
