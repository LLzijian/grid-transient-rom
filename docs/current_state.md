# 当前研究状态

任务状态：**RUNNING，阶段C公平重训**。实施用户批准的72进程小时物理闭合方程发现计划；[协议](../reports/closure_discovery_20261017_protocol.md)、[资源变更](../reports/closure_discovery_20261017_resource_amendment.md)。TASK4暂停，原冻结test关闭。

A已形成可执行首版，原S-O向量场重建误差2.22e-16；两个新增train条件通过双步长FOM核验。已知q组件例与100ms部分观测例的证据范围分别记录，后者不冒充正确公式恢复。详见[阶段A结论](../reports/closure_discovery_20261017_stage_A.md)。

B六次搜索已结束，11个合法512次端点、三个数值失败均保留。固定库与生成式均选同一八项模型，删除port es，掩码[1,1,1,0]；不能据此声称生成式优于固定库。10项替代候选满足近优及可观测分歧规则，其新增(p+q)p在当前B端点可吸收到原p二次储能和pq系数。独立只读审查同意继续原C比较，不晋升最终模型。详见[阶段B结论](../reports/closure_discovery_20261017_stage_B.md)。

当前绑定源码2e38331099ecb14bfe0d858e1cc88671fba996264bba88bd74ccfc36f017e487。repair03通过67项定向回归，不改训练数学或科学预算；只读B缓存汇总禁止fit且原模型/优化器/失败字节保持。旧f27实现及三种子固定对照保持原来源、1024次和SHA。

实际C进程：main2703 PID46164/工具会话42359；library2703由 PID14252/34842继续；alternative2702及2703由 PID46048/65518顺序继续。均按原1024更新及固定后半程衰减。详细命令、执行单与receipt见 `private_data/closure_discovery_20261017/C/active_refits_snapshot.json`。已完成十个端点：固定结构三个、library/main各前两个、alternative2701及两个删除实验；1024次端点、训练状态SHA与物理QA均核实。[紧凑进度证据](../results/closure_discovery_20261017/C/completed_endpoints_progress.json)保留全部范围与哈希。library/main相同种子的模型字节相同，不计作独立算法成功。

八项删除评分按原规则选中index3转速端口、index2边储能，两项均从同一S-O完成1024次重训（seed2701）。train 1/2/5 s风险为主候选1.7722107185、delete3为2.0254499203、delete2为1.9960797974；这是一种子、固定适配预算下的比较，不代替新增val确认或普遍必要性证明。下一动作：完成剩余四个原定端点并形成C冻结清单。确认分项输出repair04已隔离检查并归档，生产源码保持冻结；必须等所有C进程退出才应用并重新绑定D来源。D随后生成新增train24、统一追加256次、冻结全部权重后才生成新增val24。D/E科学执行尚未开始。

总授权259200进程秒；分配A43200/B57600/C86400/D43200/E14400/reserve14400。截至上述十端点快照的闭账支出 A=1990.0095387, B=15513.3064088, C=15424.0971818, D=61.9055724, E=41.0764439, reserve=403.2563691；三条在运行C进程及部分短检查尚待结算，不能将闭账余量当成全部可用预算。唯一账本 `private_data/closure_discovery_20261017/ledger/receipts.jsonl`；旧TASK2的59643.306023/60000秒单列，不转移额度。

恢复目录 `E:/03-nn_Transfer_Func/kiki`，Python `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe`。来源0102 S-O SHA256 c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238。S-O是研究候选、F-O是开发基线；train为346合格事件、含35个失稳样本，真实区间与100ms因果边界不变。

GitHub归档分支 `codex/closure-discovery-20261017`，当前已核实归档提交f25ae2d37531f15514c437309ebe6df3b8f959c7。关键设计、代码、正负结果均归档；原轨迹、缓存及中间检查点留本地，重要模型列路径与SHA。无关修改与本地HEAD a951013保持原样。工程接口见[执行说明](closure_discovery_runbook.md)；历史状态见[备份](history/current_state_before_closure_discovery_20261017.md)。
