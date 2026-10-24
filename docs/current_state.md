# 当前研究状态

任务RUNNING；cwd E:/03-nn_Transfer_Func/kiki。优先完成计算/训练效率优化，之后再开展新科学比较。原val/test关闭，TASK4暂停；40状态、100ms一次因果初始化、真实区间/失稳样本/单位/数据配额/科学及数值门槛保持，默认模型none。

当前实际作业：training_efficiency01，来源424a830197908309454f15f3393236aa59963f45，208绑定文件；独立静态审查已PASS。唯一CPU QA会话24870已退出0，9项通过、实际2次CPU更新/0GPU更新，native209项源前后/当前一致，wall44.5270161秒、JobCPU user3.53125/kernel1.71875秒。三处冻结前问题已关闭，正式修复次数0；不重跑QA。

GPU工程流水会话40421已实际启动：reference计划208更新，成功后同会话自动执行opt1的200更新；各阶段原生入口再次核同源、完整四工况与排他锁，不重复排队。初始reference worker54432/FILETIME134373316690097373，supervisor27704/FILETIME134373316685022826；后续阶段身份须读实际锁/回执，不能沿用旧PID。真实进行中更新数及wall/JobCPU待原生回执，不猜测。当前阶段源不可修改。

恢复导航：
- [有限工程协议](../reports/training_efficiency_20261024_protocol.md)、[实际Astra6返回](../reports/astra6_training_efficiency_20261024_raw.txt)、[独立静态审查](../reports/training_efficiency_20261024_static_review.md)。
- [QA证据](../results/training_efficiency_20261024/qa_summary.json)、[启动身份](../results/training_efficiency_20261024/run_started.json)。
- 作业目录private_data/closure_mechanism_20261019/P1/training_efficiency01；阶段reference/opt1、native_reference/native_opt1，active.lock；原始命令见启动记录。现有目录禁止覆盖或任意外部resume。
- 先稀疏核实际Job/失败/完成，不反复加载checkpoint；完成后核实际更新、4工况同步计时/显存、原0/2048轨迹梯度、真实32恢复、Adam/RNG及原全16事件双步长数值资格，再评价采纳。无实测前不报告加速。按用户最新要求直接检查代码，不新增定时任务：独立公式执行补丁已实现，最终CPU会话54353退出0、11项通过/29.60秒、0次optimizer/GPU更新；推理切回反传的缓存缺陷已复现、修正并保留不利证据。现有208绑定源复核一致；补丁尚未接入当前作业或完成GPU资格。[代码检查与补丁结论](../reports/code_efficiency_inspection_20261025.md)、[最终CPU证据](../results/code_efficiency_20261025/cpu_qa_final.json)。后续执行仍按现有有限工程方案，不扩成编译/精度扫描。
- 接受要求：资格全过、四工况等权几何平均加速≥1.10、单工况耗时不恶化>5%、峰值allocated不增>10%；性能不足归档不采用，不触发修复。正常最多608工程更新含条件优化二；唯一明确执行缺陷修复可按原协议追加受影响实现≤200次验证，不训练到通过。全部进程数量/时长/阶段秒/总小时额度已取消，以上为有限工作量而非进程额度。
- 所有结局归档，合格实现经独立精确范围审查后PR合main、核预期head/远端；工程整合不等于模型晋升。健康作业仅作稀疏监控；本次执行重点为直接代码优化，不以定时任务代替实现。

已关闭科学轮：horizon_pair01完成12×2048=24576实际更新、36检查点/12真实恢复，96个train评价窗口全16事件数值合格；2048时短窗口三个实例均选竞争公式、长窗口三个均选植入公式。结论限两个机制及旧train条件，不是完整候选库/独立val确认。正式wall132090.7268424秒（36小时42分钟），JobCPU user101114.828125/kernel24601.015625秒。182原生源前后/当前一致、worker46860已结束、Job0/锁清；不得重跑。[结论](../reports/horizon_pair_20261023_conclusion.md)、[完整事实](../results/horizon_pair_20261023/completion_facts.json)。独立结果审查通过，六份结果文档PR33已合main，远端1b19fa9c8034b78e1863a7f35ed81ae55d9a845c，父提交/树/六blob核同，[证明](../results/horizon_pair_20261023/main_verification.json)。

原18×512已完成6085.6472446秒、旧384测时、五例与QA06不重跑；PR30/31/32、旧失败/REJECT/未知hold与源检查点保留。账本results/closure_mechanism_20261019/cost_snapshot.json按实际wall与JobCPU分列；参考及优化作业的费用尚待回执。关键资料选择性归档codex/edge-memory-discrimination-20261021；历史大数组/模型/Adam/RNG/trace按本地路径与SHA保留。local HEADa95101337855ed3fc1f4abfd710982443d5564cb及用户未提交工作保持，不reset/强推/删证据。
