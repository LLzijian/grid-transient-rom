# 当前研究状态

任务RUNNING；cwd E:/03-nn_Transfer_Func/kiki。用户2026-10-25已授权按[隐状态与有效耦合研究建议](../reports/project_progress_research_20261025.md)开始推进。当前阶段r_memory_20261025，PREPARING：实际Astra6规划、三模型实现与独立审查已完成，13项模型、18项研究入口和15项运行控制纯检查通过，准备冻结并启动实际运行。尚无本轮科学optimizer或GPU作业。原val/test关闭，TASK4暂停，40状态、100ms一次因果初始化、真实区间/失稳样本/单位/数据条件配额与数值门槛保持；默认模型none。

旧training_efficiency01已核为INTERRUPTED_RUNTIME/UNRESOLVED。原生PID/FILETIME核查supervisor27704与worker54432均GONE，无正式完成回执；opt1未启动。持久记录精确156次更新，按冻结源顺序完成下界172、上界208，准确总数及wall/JobCPU未知。208绑定中206项一致，外部python.exe/python312.dll已改变，不能恢复旧入口。旧锁及全部输出保留，不重跑旧QA/reference，不造Job0或清理回执。[中断结论](../reports/training_efficiency_20261024_interruption.md)、[恢复事实](../results/training_efficiency_20261024/recovery_20261025.json)。未采用待GPU资格的表达式补丁，不报告加速倍数。

新科学路线：A成熟40维结构候选，B作用固定的r记忆，C同B动力学加合法历史直接初态通路；A/B为整体设计比较，B/C隔离初始化组合。root预选固定主seed6101的N1024，非按成绩择优：private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024/model.json，SHA c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5；清单results/quartic_memory_20261020/frozen_evaluation_endpoints.json。它是研究候选，见过共同train385（342稳定/43失稳）；本轮内部留组不得称未见确认。新协议、实现与Astra6采纳记录已完成；即将冻结来源，尚无新PID。

下一可执行动作：将最终协议/源码/审查归档并核远端 → freeze → 当前运行时正常/异常原生探针 → prepare → 真实8s无更新可行性 → CPU/CUDA配对及204次恢复QA → 最多96次校准 → 最多六次2048更新适配及固定评价。入口experiments/r_memory_execution_20261025.py，使用当前base Python和__PYVENV_LAUNCHER__指向.venv-gpu/Scripts/python.exe；source-commit须填核实的归档40位SHA。新输出private_data/r_memory_20261025/run01，不覆盖旧目录；历史锁只能按精确SHA/owner/创建身份再次GONE核查允许这一个保留中断锁，其他未知或活锁拒绝。单型可行性失败隔离对应模型，B/C共同选择仍需双型合格；本轮最多13080次实际optimizer调用、24次含恢复的评价队列，不设进程时间/数量限制。[协议](../reports/r_memory_20261025_protocol.md)、[实际Astra6](../reports/astra6_r_memory_20261025_raw.md)、[独立审查](../reports/r_memory_20261025_review.md)。

全部进程数量/时长/进程秒/总小时限制已取消；科学比较仍按有限模型、种子、更新与修复关闭，不训练到通过。旧累计账本及未知hold不改；旧工程实际费用未知，本轮科学更新0，后续以实际原生wall与JobCPU记账。归档分支codex/r-memory-initialization-20261025；本地HEAD a95101337855ed3fc1f4abfd710982443d5564cb及其他未提交工作保持，不reset/强推/删证据。

已关闭horizon_pair01：24576实际更新，36检查点/12真实恢复，train16事件的96个窗口评价均数值合格；短窗三实例选竞争机制，长窗三实例选植入机制，限两机制及既有合成train。[结论](../reports/horizon_pair_20261023_conclusion.md)、[完整事实](../results/horizon_pair_20261023/completion_facts.json)。正式wall132090.7268424秒，JobCPU user101114.828125/kernel24601.015625秒；PR33已合main并核远端1b19fa9c8034b78e1863a7f35ed81ae55d9a845c。原18×512、旧测时、五例、QA06、所有失败及检查点保留，不重跑。
