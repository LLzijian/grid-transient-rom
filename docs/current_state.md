# 当前研究状态

任务状态：**COMPLETED，仅指本次有限A—E实施、执行与记录已完成**。物理结构恢复、生成式搜索优势和整体确认改善尚未成立；不表示项目科学路线全部完成。所有科学进程已退出，没有后台或计划任务。原test关闭，TASK4保持暂停，未改变部署默认模型。

[最终报告](../reports/closure_discovery_20261017_final.md)汇总算法、实验与证据范围；[原协议](../reports/closure_discovery_20261017_protocol.md)及[资源变更](../reports/closure_discovery_20261017_resource_amendment.md)保留。A实现与成本检查后，搜索深度前瞻锁定一轮。B完成两种方法各三个搜索种子；C完成四类结构各三个1024次重训和两项完整删除；D完成24新增train、24锁定新增val、14×256次适配与冻结确认；E完成显式方程、独立运行和数学检查。

关键结果：main/library在三个相同种子上模型字节与评价相同，不支持生成式方法额外收益。A的部分观测问题未恢复植入饱和项。新增val24真实标签20稳定/4失稳，1个失稳样本只覆盖0.07 s不能100 ms初始化，保留不可评价；23条共同进入全部模型评价。20条完整5 s稳定事件上main2701风险2.43945313、fixed2701为2.45231318，差−0.01286006、95%区间[−0.03361368,−0.00032950]；仅为合格子集诊断，完整确认按原规则保持未决。main漏检1/3、fixed漏检0/3，均假失步1/20。详见[阶段D](../reports/closure_discovery_20261017_stage_D.md)。

14条最终端点对应11个不同模型，全部为研究候选。主模型存在去共同角后的正实部局部模态；仅单种子delete2通过39维严格Hurwitz线性检查，没有非线性稳定域证书。delete2在新val的5 s风险较低但短时与频率分项有损害，没有据val重新选择。独立审查及回应见D/E结果目录。[阶段E](../reports/closure_discovery_20261017_stage_E.md)及[主方程包](../results/closure_discovery_20261017/E/packages/main_2701/README.md)。

最终绑定源码b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc，30文件完整性通过；14个冻结模型SHA及逐事件误差重建一致。确认冻结SHA6bb7d43997222ffa9c7bb47728c0117d6b8b25712f5df0141f38f06fca78751d。源为0102 S-O step650，SHA c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238。S-O为研究候选，F-O为开发基线。

费用快照2026-10-17T02:57:34.196230+00:00：总上限259200秒（72进程小时），已核算62639.270247秒（17.3998小时），未用196560.729753秒。A/B/C/D/E/reserve分别为1990.009539/15527.135760/32729.761889/11457.973091/225.498610/708.891358秒；科学进程已全部计入，快照后的归档核验按实际回执继续计费。[费用与口径](../results/closure_discovery_20261017/final_cost_summary.json)，唯一账本private_data/closure_discovery_20261017/ledger/receipts.jsonl。旧TASK2的59643.306023/60000秒单列、不转移。剩余额度不自动授权新搜索或新研究。

停止原因：登记的有限A—E执行及记录已结束，保留负面和未决结论；没有以预算耗尽或稳定性成功为由结束。下一可执行动作是复核既有不可变结果或运行已导出包；进一步科学实验需要新的前瞻问题、预算与授权，不重放当前train/val确认。

恢复目录E:/03-nn_Transfer_Func/kiki；Python E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe；检查点和命令见[执行说明](closure_discovery_runbook.md)、D冻结及E交付清单。active_job.json中pids为空。关键设计、代码、正负与未决结果归档分支codex/closure-discovery-20261017；原轨迹、缓存、重复检查点和大矩阵留本地，重要路径与SHA公开。无关修改、本地HEAD a951013和用户index保持。

最终交付提交8ea67edbe29122e84f3fc33427612e5269173ee5已核实远端分支及全部17份交付文件字节；归档核验与费用快照随本次收尾更新保存。
