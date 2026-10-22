# 当前研究状态

任务RUNNING，cwd E:/03-nn_Transfer_Func/kiki。人类已批准“允许修复、验收并继续训练”，本轮范围固定五例本机检查+一次完整QA06；通过即原12组GPU，不再等待同一许可。真实gpt-6-astra新规划及root采纳已保存[方案](../reports/astra6_spawn_chain_20261022.md)，[协议](../reports/spawn_chain_repair_20261022_protocol.md)。当前修复源码和三入口连接已实现，非实现者最终静态审查进行中；五例/QA06/科学训练均尚未执行，无活动科学PID/队列或全局锁。

五例normal父子、父子孙、管道、真正孤儿、监督进程退出各一次，任一不符合即结束，不修后重跑；全合格才QA06一次，不追加QA07。各次Python派生显式DETACHED；Job保留严格活动成员清理/FAILED，仅增加PID/FILETIME/映像取证。监督退出由外层保留具体child process句柄等待真实终止；外层不得代为清理后声称内层句柄关闭清理。旧A2源码/选择及QA01–05全部原件保留；新运行环境选择位于spawn_chain20261022/runtime_selection.json，不能套用旧A2。

QA05历史：133通过2失败，真实源5b7722c0cf66b3eb5313d9f4efd6d2f007a551e4，native SHA34283382ab6da9e5901062986bdfacd3301a1379cf3d423ef576e0fb3f4dd548；80.681267秒wall、CPU user10.484375/kernel19.3125，Job最终0、源106项前后相同、锁清。原26项路径错误已消除；剩余为测试正则和正常Job内额外活动成员，尚无映像，不能推断原因。[结论](../reports/qa05_recheck_20261022_conclusion.md)、[摘要](../results/rational_execution_20261021/qa05_summary.json)。历史失败与费用不改。

下一实际动作：静态审查收尾、源与决定归档冻结→一次五例→一次QA06→通过自动0_rational→0_tanh→1_rational→1_tanh，各seed6301/6302/6303、各512，共6144；0/32/128/512保存、32真实恢复、最多一次非择优中断恢复；12模型冻结后一次合成val。训练当前0，模型采用none，main默认不变，L1/L2仍未完成。原项目val/test关闭、TASK4暂停、40状态/100ms一次初始化、真实区间/失稳样本/数值及模型门槛不变；旧18×5126085.6472446与384测时不重跑，数据条件配额不扩。

进程数量/时长/阶段/总额度均取消，实际wall/Job CPU继续累计，未知支持费用不伪称精确。费用账本results/closure_mechanism_20261019/cost_snapshot.json；历史hold/REJECT保留。启动入口private_data/closure_mechanism_20261019/P1/rational_execution01/{make_binding.py,launch.py}，旧失败和检查点原路径保留。归档分支codex/edge-memory-discrimination-20261021，最近已核760caa5fbbc388299ad2d8d7f061c8e253b965a4；main仍ad1b48a3b701cfaf951d0e96067d5fb537e74c90。无本轮PR合并；不reset用户工作、不强推、不删证据。heartbeat保持。
