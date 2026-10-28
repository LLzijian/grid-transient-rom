# 当前研究状态

任务 **RUNNING**；cwd `E:/03-nn_Transfer_Func/kiki`。目标为物理系统的结构发现与识别，按算法、关键事件、预测与同精度成本共同评价。原val/test关闭，TASK4暂停，默认模型none，可用模型档案保持。科学候选与可用模型分开保留。

**当前阶段：force_graph_20261028，RUNNING。** 实际gpt-6-astra方案已由root采纳并落实事前修订，见[原文](../reports/astra6_force_memory_20261028_raw.md)及[协议](../reports/force_graph_20261028_protocol.md)。gpt-6.1-sol负责实现与执行，Luna整理有界事实。比较共同因果积分初态下的局部记忆L、静态耦合S与传播记忆M；40状态、100ms初始化及固定物理骨架保持。主来源A/25101 step000512在R0结果前固定，备用N6101 step001024。模型29项（含3项真实CUDA零更新）、study25项、controller21项检查与独立复核已完成。校准失败隔离在真实运行前修订并归档。完整pipeline真实启动，exec session16410；prepare已完成，最近核实阶段qualify。supervisor PID5432/FILETIME134376602622344859，资格worker PID28892/FILETIME134376602821416498；后续阶段以实际ownership文件为准。binding SHA ddcd61c0345c1db8dc5f4e3ed5c5cddb794620bcd9c84b12bec341d681027b16。临时heartbeat实际ID `kiki-force-graph`，20分钟，供长运行保活；完成/需行动即删除。

**当前命令与下一动作**：6.1已实际运行 `experiments/force_graph_execution_20261028.py run --source-commit fc5e4ea9c434ad442cfd1d99ebf6bc7e22600875`；driver使用direct_base_venv。等待真实资格/QA结果，合法后由同一可执行pipeline继续，无需定时器触发下一阶段。输出 `private_data/force_graph_20261028/study01`；流程自动执行prepare、零更新积分精度/成本资格、真实CUDA保存恢复QA、有限校准、正式比较/按曲线条件延长/备选来源及噪声评价、summary。总实际optimizer调用上界1304（960正式、至多48校准、108QA、108仅一次实现修复预留、80外部中断重放预留）；预留不自动执行，不增补模型或参数搜索。启动前16个新文件逐blob与远端head均核实；65项旧Python源码未变，新AST闭包无额外未归档依赖。

**旧阶段：INTERRUPTED_RUNTIME，停止自动续跑。** driver51852/FILETIME134374012174040563与worker61692/FILETIME134374012197238172均GONE；正式退出原因及完整成本UNKNOWN，没有正式fit原生回执。仅A/25101运行，1165次实际完成更新，持久检查点1152，后13次无可恢复模型/Adam/RNG；其余五组未运行，不判科学失败。旧锁SHA `d764e2c272182acc6c54dda0a9edf1309c3475b1e79916305652b2adfeb123bf` 保留。恢复时87项全部MATCH是当时快照；其后AGENTS/engineering前瞻规则已按用户要求修改，不重写旧绑定。见[恢复事实](../results/r_memory_20261025/recovery_20261028.json)。

**R0冻结评价COMPLETED，0更新。** 固定A1152模型SHA `5e8bba7c406d12ec9745664c7aeb8cd8d1aebe29f3c0257b9ef13e0109f12aac`，输出 `private_data/r_memory_20261028/recovery_eval01`，原生退出0、Job空且清理通过。共同train385稳定342的8s风险由A0的1.620774到A1152的1.186944，失稳43的8s漏判由5到7，稳定误报由4到1；风险不是RMSE。N/A0、A512、A1152按互补能力保留为科学候选，未采用为默认。来源见过全部385，fit297及本轮不追加更新88均为train发展证据。详见[汇总](../results/r_memory_20261025/recovery_evaluation_20261028_summary.json)及[结论](../reports/r_memory_20261028_conclusion.md)。

**费用与归档。** 旧阶段实际调用1466次、完整更新1465次；11份关闭Job的部分成本user622.65625s、kernel275.09375s、逐Job wall合计854.0326571000041s，缺中断正式fit成本。R0单独wall154.5262543s、user118.09375s、kernel9.0625s；不能串成完整阶段总成本。无进程数量/时长/CPU小时额度上限；每轮科学工作量仍有限。归档分支 `codex/force-memory-20261028`；本地HEAD `a95101337855ed3fc1f4abfd710982443d5564cb` 及无关改动保留。实际冻结执行来源 `fc5e4ea9c434ad442cfd1d99ebf6bc7e22600875`，远端逐blob核验见[source_verification](../results/force_graph_20261028/source_verification.json)。新长作业需要时创建临时heartbeat，完成/需行动时删除并同次回标准loop。

原roadmap见[TASK2 after101](TASK2_after_loop_101.md)，执行见[engineering_loop](engineering_loop.md)及[10月28日补充规则](research_execution_20261028.md)。旧training_efficiency01及horizon_pair01的中断/关闭分类、锁、证据及已合PR33保持；新研究不改写历史采用与失败结论。
