# 当前研究状态

任务 **RUNNING**；cwd `E:/03-nn_Transfer_Func/kiki`。核心是物理系统的结构发现与识别；预测、关键事件、条件辨识、复杂度和同精度成本共同评价。原val/test关闭，TASK4暂停；默认none，可用模型及原接受范围保持。

**force_graph_20261028正式运行中。** L96、S192、M192已完整评价共同train385，必要数值行均合格。稳定342的5/8s风险分别为L .858630/.579123、S .732162/.516601、M .685761/.503166；失稳43的8s漏判分别15、10、8，误报均0。各为登记适配端点，S到上限仍改善，不能写充分优化。风险不是RMSE。全部是已见train的发展证据；fit297决策，不追加更新88仅报告。

既定fit-only选择M为结构、S为对照，备用来源N6101/种子28102已自动启动，最近核实N_M第87次更新。之后仍须完成N_S及最多五端点的固定噪声报告，不跳过分支。此前108次真实CUDA保存恢复QA、48次校准全部闭合；三模式均选倍率3，校准状态丢弃。fast未全过原精度门槛，统一真正RK4。

实际命令 `experiments/force_graph_execution_20261028.py run --source-commit fc5e4ea9c434ad442cfd1d99ebf6bc7e22600875`，exec session16410；supervisor PID5432/FILETIME134376602622344859，study worker PID32588/FILETIME134376611306402684。输出 `private_data/force_graph_20261028/study01`，binding SHA ddcd61c0345c1db8dc5f4e3ed5c5cddb794620bcd9c84b12bec341d681027b16。最近已核新轮至少723次真实完整调用（QA108+cal48+主源480+N_M87），不是最终总计。study原生总成本待关闭回执；无进程数量/时长/CPU小时上限，科学调用上界1304不变。

**下一动作**：6.1继续已登记pipeline并稀疏检查；root在全部端点与原生清理闭合后评价、独立挑战、选择性归档/决定集成，接续合法新科学阶段。临时heartbeat实际ID `kiki-force-graph` ACTIVE/20分钟，仅保活；完成或需行动即删除，同次回标准loop。

**后续准备，不是新作业。** 实际Astra复议不执行两图筛查：数据共享同一故障后系统，条件选图会增加1bit模式。推荐共同40D内单gamma的p-r互易交换；独立数学挑战通过，root仅采纳独立新模块和CPU零optimizer准备。F来源待当前轮完整闭合再按fit规则选定；新轮不提前训练。见[设计补充](../reports/astra6_next_identification_20261028_addendum.md)及[独立挑战与root决定](../reports/force_exchange_20261028_design_review.md)。

**恢复与档案。** 旧r_memory停止续跑，状态INTERRUPTED_RUNTIME；1165次完整更新只保存至1152，退出原因及正式总成本UNKNOWN，旧锁/证据保持。R0已完成1152冻结评价，0更新；原N/A0、A512、A1152保留科学候选，未替换可用模型。旧阶段调用1466、完整1465，已关闭11份Job部分成本不含中断正式fit；R0单独wall154.5262543s、user118.09375s、kernel9.0625s。见[R0结论](../reports/r_memory_20261028_conclusion.md)和[原恢复](../results/r_memory_20261025/recovery_20261028.json)。

归档 `codex/force-memory-20261028`，冻结执行源已逐blob/远端head核实，见[source_verification](../results/force_graph_20261028/source_verification.json)。本地HEAD a95101337855ed3fc1f4abfd710982443d5564cb与无关改动不动。规则PR34已独立审查并合入远端main 6f1d97deec28cece1d93a1b7e29dfe217de881a4；科学候选尚未采用。导航：[本轮协议](../reports/force_graph_20261028_protocol.md)、[after101](TASK2_after_loop_101.md)、[执行规则](engineering_loop.md)。旧关闭/失败分类不重写。