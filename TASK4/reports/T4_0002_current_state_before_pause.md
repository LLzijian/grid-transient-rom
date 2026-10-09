# TASK4 current state

- mission_status: RUNNING；用户最新指令“完成当前实验后暂停”，闭合T4_0002后暂停，不开下一轮。
- active_roadmap: TASK4.md，route_revision=1，pending_direction=none。
- authorization: 用户2026-09-30明确开始并给出100000累计进程墙钟秒；2026-10-07及10-08“请继续推进”。预算未重开，不继承TASK2。
- active_loop: T4_0002，reports/T4_0002_idea.md；55000预留不是花费。
- active_work: T4_0002的N800、七函数C、最终S800均完成，两个端点已由reports/T4_0002_frozen_endpoints.json冻结；正在顺序执行两新端点的固定开发评价。历史long400+800仅计1200，short/anchor历史分列。权威活动run/PID见state/research.json；闭合本轮后按用户指令暂停。
- current_TRAIN_evidence: N200/400/800固定12事件监控mean=.5040355/.5134606/.5229479；S200/400/800=.4738826/.5438842/.5669194，均未双步长验证、非开发证据。固定终点未按监控选择；保留不改善与大梯度记录。N/S均800/800、346/346 TRAIN、head未优化/使用/部署、无shooting reset。见reports/T4_0002_progress.md。
- current_C: 七函数C完成，run_aad8c1d9f431退出0/实际592.671秒，C SHA5b66422ee5f36470c14d4450b8f9f685d32ab5e971332a2ca872cf834143594e。seed419/adapt16/max2048、N源前后SHA一致，真实argv及返回值均保存launch_receipts。六函数degree2/port degree3，两个储能unique degree2合格energy_only。三步完整自主成本探针已完成，正式D直接从C起步，C和探针均不冒称最终D。
- current_S: 正式S_final run_c5a7c6f6fc5e正常完成固定800步，外层实际12023.062秒、346/346 TRAIN，SHA4cdb946b627626894997adf6c6c21906b0776e09a26dc368f3b37c5970914138。从固定C重置Adam，energy_only仅两H、全ODE+GRU、aux/teacher/reset0。七函数显式独立运行包已导出，freeze run_97152f1cc4cb成功；当前读取登记开发，不再改变端点。
- accepted_prediction_model: none；accepted_candidate与active_candidate=null。
- research_candidates: T4_0001/S_final与A_long仅互补retain，均不可部署；N/U完整实验资产保留。
- accepted_code_ref: logs/run_6e4e003652b7/source.zip；旧269/25/31/16/31及新96来源冻结/协议保护QA和独立只读复核通过。两次QA失败/修复保留，科学模型未晋升。
- code_scope: trace/原点项等价执行、默认off二次执行器、全状态失败隔离、来源冻结、有限缓存复算、显式公式/区间/结构、初态中心检查、路径契约与C来源/头部作用报告。二次执行器仅合格degree2函数可用，不承诺逐位或所有输入溢出边界相同。
- data: ../kiki严格只读；TRAIN原369/可覆盖346/可覆盖失稳35/时延不可覆盖23；开发原117/可覆盖107，失稳原22/可覆盖12/时延不可覆盖10。TEST载荷未读，0075/0077不自动并入，无新FOM生成。
- runtime: E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe，CPU float64；默认监管Python+TASK4/.runtime；不修改复用环境/KIKI。
- process_policy: 最多一个受管重任务、8GiB；当前用户提供AGENTS的外层查询240秒优先于旧本地900秒说明。普通检查仅状态/最新日志，不载入模型或重算指标；对外合并里程碑，N新阶段200/400/800。
- budget: T4_0001 CLOSED，实际32413.790932秒，首轮94000未用61586.209068归父池；开T4_0002前父池67586.209068。启动状态辅助导入失败与更正4.257686秒另计本轮；当前成本以state/research.json权威更新。
- first_loop_report: reports/T4_0001_closure.md；完整固定成绩单artifacts/T4_0001/full_comparison/capability_table.md及comparison.json。全部5端点来源/队列/尺度/双步长契约一致、无数值未决/积分失败；第二代理只读复核通过。均为开发证据，非新组独立确认。
- first_loop_prediction: N相对运动改善但5/8s common/p损害；S mean/common/p改善但δ pooled、严重前缀及漏检损害；A/U各检出11/12，N/M3各10/12，S9/12，所有假LOS0。A/U唯一漏检expansion_0155；N/M3另pilot_0155；S再expansion_0189。S另stable pilot_0232窗口未定。window_settled仅窗口筛查，非证书。
- first_loop_formula: 全七模块/34共享标量输出/544非零系数，810 dense槽、2772共享多项式AST节点（不是整RHS完全展开）；GRU4194、coupling45另列。独立NumPy公式可计算，保留一次神经初始化；实际公式、50行因式RHS、复杂度在S_export/S_complexity。
- first_loop_structure: N/S各856预选状态的解析/独立autograd梯度、全局J/R/端口重组及完整场通过；五端点各同一12预选事件有限缓存独立公式复算通过。root运行数值，第二代理仅人员独立只读复核；不是第二人重跑全队列/分类库。
- first_loop_region: 同一最终S连续49维商空间的8登记半径完成，至.003的5个内含椭球通过，.01/.03/.1充分条件未决；数学ROM、区间库条件依赖，非整个盒/Poisson/全局/FOM/50维点稳定。实际107初始化中心0 inside/107 outside/0invalid；未获内界覆盖不等于失稳，初始化误差集合未覆盖。新S权重必须重算，不能继承旧证书。
- first_loop_modes: N/S原标准有效0/13，保留完整原门槛；附加结果不否决L1。
- first_loop_cost: 固定CPU1/float64 5s含一次GRU初始化+积分/finite检查，中位N/S单事件18.2189/16.7404秒，B8整批36.7968/29.5508秒；加载/history/trace另列、观察.1s另列。16计时trial+1S Torch参考全部完成，S的8次物理parity通过，N无独立parity；full-state差为诊断非阈值门。runtime-only，无同精度或FOM提速资格。
- engineering_cost_probe: 固定S400首stable/unstable TRAIN面板三模式各一步，梯度/Adam14参数全部一致；reference/energy_only/all_poly 31.788/21.321/27.914秒仅一次顺序工程成本，非部署或精度声明。后续S仅H合格时优先energy_only。
- historical_limits: P_anchor run_c0e9c37eed40当时独立source.zip缺口如实保留，不能端到端精确重放；旧基线热启动不当独立新组确认。P-N初慢eager尝试保留2批缓存，无最终JSON；失败成本不扣除。
- next_required: 完成两个冻结端点的固定开发评价、有限独立复算、最终ODE数学与运行成本证据、完整能力表及决定，然后PAUSED；不启动新一轮。会话中断按INTERRUPTED_RUNTIME记录并实际恢复。

用户最新暂停指令优先：完成当前T4_0002的S800、冻结、登记评价/有限复核及闭合后暂停，不执行新一轮。普通工程修复及科学阴性结果不自动触发改向审批。
