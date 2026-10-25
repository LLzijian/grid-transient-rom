# 当前研究状态

任务RUNNING；cwd E:/03-nn_Transfer_Func/kiki。用户2026-10-25已授权按研究建议推进。当前r_memory_20261025唯一纯实现修复及只读复核已完成，REPAIR01_PREPARING：三型真实8s前向/反传与CPU/CUDA配对通过；首A Adam调用后，优化器状态有限性检查混用CPU计数/CUDA动量而失败。实际optimizer调用1，完成one_update核验0；校准/正式更新0，当前无新运行作业。原val/test关闭，TASK4暂停，40状态、100ms一次因果初始化及数值门槛保持；默认模型none。

旧training_efficiency01已核为INTERRUPTED_RUNTIME/UNRESOLVED。原生PID/FILETIME核查supervisor27704与worker54432均GONE，无正式完成回执；opt1未启动。持久记录精确156次更新，按冻结源顺序完成下界172、上界208，准确总数及wall/JobCPU未知。208绑定中206项一致，外部python.exe/python312.dll已改变，不能恢复旧入口。旧锁及全部输出保留，不重跑旧QA/reference，不造Job0或清理回执。[中断结论](../reports/training_efficiency_20261024_interruption.md)、[恢复事实](../results/training_efficiency_20261024/recovery_20261025.json)。未采用待GPU资格的表达式补丁，不报告加速倍数。

新科学路线：A成熟40维结构候选，B作用固定的r记忆，C同B动力学加合法历史直接初态通路；A/B为整体设计比较，B/C隔离初始化组合。root预选固定主seed6101的N1024，非按成绩择优：private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024/model.json，SHA c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5；清单results/quartic_memory_20261020/frozen_evaluation_endpoints.json。它是研究候选，见过共同train385（342稳定/43失稳）；本轮内部留组不得称未见确认。新协议、实现与Astra6采纳记录已完成；冻结来源与当前87项源码/运行时身份已保存。

下一可执行动作：归档新SHA → 新目录private_data/r_memory_20261025/run01_repair01 freeze → 正常/异常探针 → prepare → 8s无更新检查 → CPU/CUDA及204次恢复QA → 原最多96次校准 → 六次2048正式更新及固定评价。旧run01、原绑定59f573b01b08c1ae221f4022c2100d411088161b、失败日志与旧锁完整保留；无旧job恢复或原地覆盖。入口experiments/r_memory_execution_20261025.py，当前base Python及__PYVENV_LAUNCHER__指向.venv-gpu/Scripts/python.exe，--source-commit填新核实40位归档SHA。[协议](../reports/r_memory_20261025_protocol.md)、[唯一修复](../reports/r_memory_20261025_repair01.md)、[实际Astra6](../reports/astra6_r_memory_20261025_raw.md)、[审查](../reports/r_memory_20261025_review.md)。

全部进程数量/时长/进程秒/总小时限制已取消；科学比较仍按有限模型、种子、更新与修复关闭，不训练到通过。旧累计账本及未知hold不改；旧工程实际费用未知，本轮实际optimizer调用1、合格更新0，后续以实际原生wall与JobCPU记账。归档分支codex/r-memory-initialization-20261025；本地HEAD a95101337855ed3fc1f4abfd710982443d5564cb及其他未提交工作保持，不reset/强推/删证据。

已关闭horizon_pair01：24576实际更新，36检查点/12真实恢复，train16事件的96个窗口评价均数值合格；短窗三实例选竞争机制，长窗三实例选植入机制，限两机制及既有合成train。[结论](../reports/horizon_pair_20261023_conclusion.md)、[完整事实](../results/horizon_pair_20261023/completion_facts.json)。正式wall132090.7268424秒，JobCPU user101114.828125/kernel24601.015625秒；PR33已合main并核远端1b19fa9c8034b78e1863a7f35ed81ae55d9a845c。原18×512、旧测时、五例、QA06、所有失败及检查点保留，不重跑。
