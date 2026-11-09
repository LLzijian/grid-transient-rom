# 固定状态下的物理闭合方程发现：执行协议

2026-10-17，用户批准实施完整算法和72进程小时研究计划。起始本地提交为a951013；工作区已有其他未提交修改，保持原样。该协议前瞻接续TASK2，不恢复暂停的TASK4，不重判0102–0106，不开放test。

## 问题与科学产物

从100 ms因果观测出发，在十机40维状态及已核实物理骨架下，自动发现闭合表达式及其支持。产物是有效降阶方程、一次初始化器、机制删除后重适配证据与适用范围。H/J/R分解及隐藏状态不主张唯一真实物理含义。

状态为delta、pi=M*omega、p、q，每机一个q；对外仍为delta/omega/p/q。固定惯性、物理图、运动学、单位及输入权限。所有骨架、闭合系数和GRU共同学习。储能不含pi；J/R仅使用既有常数方向，R系数非负，所有端口纳入完整功率账。禁止未来部署输入、途中重编码、补造失稳轨迹或移除困难事件。

## 表达式与算法

节点储能使用归一化p和q；边储能使用两端p/q及中心化角差sin/cos，按端点交换对称化；端口使用w/p/q和既有邻域汇总。加、乘、平方/立方/四次方，端口允许一次tanh且尺度限0.5/1/2。深度<=3，运算节点<=12，储能多项式次数<=4，端口<=3，最多12个储能/端口项。禁止除法、指数和任意神经RHS。H消去平衡点常数和一阶项，端口归零。q沿用原数值坐标与固定尺度；不逐候选重估尺度。保留q二次基项及原J/R四个方向的开关/幅度，新增项系数从零开始。先要求精确重建S-O。

固定库与生成式方法使用相同选择粒度、拟合程序和计数。每搜索保留两个候选，最多三轮，每轮六候选，覆盖增/删/替换。train短程方向梯度只排列提案。全部候选32次更新，前两名128，再两名512；只比较相同适配等级。搜索种子1701/1702/1703，重训种子2701/2702/2703。误差-复杂度非劣保留；完整train风险一个分组标准误差内优先最简单者；替代候选由可观测轨迹分歧选择，先排除q的符号/尺度重参数化。

目标沿用原自主风险、局部轨迹、隐藏一致性、转矩辅助，权重10/1/0.1/0.1；去除原系数L1，复杂度在外层计算。原1/2/5秒时域与事件采样不变；Adam骨架/闭合/GRU率为0.01/0.001/0.0003，clip100，float64/CPU单线程。正式重训1024次，后512次学习率乘0.3；均从同一登记S-O来源重建，新增表达式从零初始化，既有表达式继承原系数。完整train监控仅登记节点，梯度和裁剪记录保留。

## 对照与证据

阶段A：三个语法内受控问题（节点、边、饱和）及一个语法/状态之外问题，无噪声与1%噪声；前两类显露隐藏坐标以检查公式恢复，第三类仅100ms观测。将同坐标恢复与未知隐藏坐标的可观测动力学恢复分开。真值由固定种子4801生成并在运行前导出，评价真值不得泄漏到提案选择。

阶段B：固定库选择与表达式生成各三个完整搜索种子。阶段C：固定原结构、固定库优胜、生成式主候选、预测分歧替代各三个重训种子；主候选最多两组机制删除后各1024更新（2701，同源完整对照）。历史F-O与TASK4只作描述参照。

阶段D：在合法train母线生成最多24个新增train辨别条件、24个新增val确认条件。时长为原网格相邻中点，电抗为0.001/0.01及几何中点；排除已有条件。新增val在查看候选成绩前用种子3701固定，短/中/长各8条（<0.15、0.15–0.3、>0.3秒），均衡母线；不按结果抽换。新增train依据已有train历史预测分歧排序邻近条件，不能读取新增val未来。新train允许统一追加256次更新，之后全部公式/系数/GRU、主候选和比较冻结。新增val一次评价，不回流选择。现有故障仅改变初态，不称跨机制环境；本轮不做PSS/拓扑/运行点变更。

预测指标：稳定完整5秒事件的既有归一化轨迹风险为主要指标；1/2/8秒、共同频率、p、失稳样本真实时间范围及分类分开。按母线分组、事件为单位配对bootstrap2000次（seed5701），95%风险差区间支持改善才作相应主要改进声明；损害与不确定结果保留。val使用仍是开发证据；新val是本轮冻结后确认，原test关闭。延迟不可覆盖者计数，不额外剔除事件。

数值资格沿用原真实评分范围的双步长、归一化差异0.001及原分类规则；不合格范围的正式汇总保持未决，保留所有行。能量恒等式不是正定性或稳定域；最终方程才可作条件性的局部分析。模型保持research candidate身份，未经独立挑战不得晋升。

## 资源、失败与阶段决定

新增独立授权259200进程秒；旧TASK2已花59643.306023/60000秒单列，旧余额不转入本研究。A43200、B86400、C57600、D43200、E14400、全局预留14400。失败/重试/数值核验计费，子进程与父等待不重复计费，并行按进程分别累加。预留只修实现/数值故障，不扩科学分支，不跨阶段自动挪预算。

A用实测代表性模型每更新成本和固定评价成本乘1.25，测算完整B/C/D/E。B每轮更新数=6*32+2*96+2*384=1152，共6次完整搜索；C十四次1024更新；D十四次256更新及最多48条件的双步长FOM和终点评价。按深度3/2/1选择完整预算可容纳档，看到结果前冻结。最低完整档无法容纳则不启动不完整对照，不削种子或适配次数，不称科学模型族失败。

实现错误先隔离相应证据，保留失败记录；可修复时用登记预算修正后重新做必要QA。数值失败不等同物理失稳；优化仍改善、适配不足、缺少替代候选或数据不可区分时结论保持相应未决。最多两个连续代理诊断后回到自主试验。阶段结束记录完整成果、消耗和下一决定，预算/运行中断必须给出恢复点，不能称任务完成。

## 工程与归档

正式搜索前的前瞻资源变更见[资源分配变更](closure_discovery_20261017_resource_amendment.md)：总额仍为72小时，阶段分配改为A12/B16/C24/D12/E4/reserve4，搜索固定一轮。全部对照、三种子和适配次数不变。该变更基于复杂边项及D完整执行路径的实测成本，适用未来执行，不修改既有账目。

最终实现细化：表达式在加乘交换规范化之外，合并重复因子，并用精确多项式展开、边对称化及物理中心化检查零项和非零标量倍重复；该检查不修改运行方程，也不删去正值骨架无法覆盖的有符号平方项。每个端口表达式可各含一次tanh。一个标准误差按删除一个故障母线的风险重新计算，不把普通均值误差的不确定性套给尾部风险。替代模型的1秒train可观测分歧使用既有固定物理尺度，归一化均方差须大于1e-4；相同尺度用于D的条件排序。D的24个新增train同样按短、中、长持续时间各八条选择。新FOM核对固定系统指纹及每次case/版本来源；已完成生成、冻结和评价按哈希复用，不能用恢复操作重采样或覆盖冻结。运行源码、依赖版本、train身份在B前绑定并供后续恢复核对。

实施细化（正式搜索前登记）：采样严格使用0102的每批12事件（10个分层抽样+2个循环覆盖），覆盖游标随优化器/RNG保存。新协议保留前100次2秒、随后1/2/5秒循环，不继承旧执行器800次上限。预算按实际各时域次数加总实测成本再乘1.25；最慢单步成本同时保留，不能把所有更新都错误计成5秒积分。A先对两个已锁定候选新增train条件做双步长FOM成本测量；不读取新增val，不把这些未来标签用于结构搜索。两个预跑条件保留在本地，D不因其结果改变确认条件。短程提案按风险减去0.001倍编辑参数梯度平方和排列，仅供六候选筛选。固定库与生成式均可切换四个既有J/R机制方向；生成式必须实际编辑父表达式子树。四候选最终重训和两删除组的完整更新数不变。

独立只读实现审查识别三项必须修复的问题：生产提案不能只是扩展固定字典；1024步不能调用旧800步日程；边储能必须在交换对称化后按完整物理变量中心化。修复及直接回归测试在正式搜索前完成。受控前两例是显露q的函数项辨识后自主积分验证，第三例是盲候选的GRU/系数联合轨迹拟合；这两类证据不冒充完整生产搜索的独立确认。

新专用模型/搜索入口，不修改旧协议执行器。保留encode/field/rollout/export/from_export及状态顺序；表达式、系数、物理尺度、初始化器和来源哈希进入导出。验证旧场精确重建、公式/梯度/独立NumPy一致、交换/耗散/功率恒等式、因果/数据边界、保存恢复和预算。

四个合法J/R方向的幅度始终保留为可训练坐标；未启用方向的梯度为零，启用后才能按既定学习率适配。参数可训练标志随模型保存和恢复，避免优化器组漂移。D的256次新增train适配以C模型权重为来源，统一新建Adam及事件覆盖顺序，使用原三类初始学习率与从头计数的时域日程；不沿用不同搜索的优化器历史，也不按val调参。

每阶段先登记后执行，重要设计、代码、正负结果、复现命令和资源记录提交并推送GitHub。缓存/原轨迹/中间检查点留在原存储；关键大文件列路径和SHA256。只暂存本任务文件，保留无关修改。独立审查聚焦科学假设、混淆与声明范围。

## 实现修复03与来源桥接（后续C/D执行前）

两条f27绑定的B/C进程均已退出后实施已前瞻登记的修复03。结构身份包含表达式支持与J/R方向掩码；高等级数值失败保留原score32两名，不补第三名；C已失败适配不因恢复而自动重试。完整检查点恢复只读核实，避免把旧端点的权重和来源改写成新实现。三项旧C固定对照保持原1024步端点字节、SHA和f27来源；后续C从同一S-O重新拟合，方程、损失、学习率、采样、种子及适配机会均不变。源码差异和恢复等价回归随修复归档。

A已预跑的两个train条件计入D的24个新增train（短、中持续时间各一条）；其余22条按原train预测分歧和分层规则选择，短/中/长补足7/7/8，总数仍24。新增val24的预锁定条件不变，因此本轮最多48个不同故障条件。A原始数据保留，D统一生成时允许对相同两个条件再次做双步长数值核验，重复运行计费但不产生额外条件。条件选择身份同时绑定模型名与种子，防止三种子来源相互覆盖。

B首次f27运行观察到三个数值失败（其中一个在完成132次更新后、尝试133次时非有限）；这些失败和原始选择均保留。修复后的B只用原始等级缓存重新汇总，禁止追加任何拟合更新。表达式字符串变化或库外AST不单独作为新函数族、新物理机制或潜坐标唯一性的证据。

## Implementation repair04: confirmation components and new-train discrimination

This prospective amendment applies to D and E after all C processes have exited.
C remains bound to source identity
`2e38331099ecb14bfe0d858e1cc88671fba996264bba88bd74ccfc36f017e487`;
its weights, optimizer states, classifications and evidence are preserved.

The confirmation output adds separate common-frequency and mechanical-power
components at 1/2/5/8 s and over the genuine available interval capped at 8 s.
For n future points, prediction indices 1:n+1 are paired with x[11:11+n].
Common frequency is mass weighted, with RMSE in rad/s and normalized MSE using
the frozen train frequency scale. Mechanical power is Pm minus its equilibrium
value, with RMSE in p.u. on the 100 MVA system base and normalized MSE using
the frozen per-machine train p scales. Each component retains physical SSE,
scalar count, availability and finite-status fields. Unavailable or nonfinite
values remain explicit. Existing evaluation caches missing the added component
field are rejected without overwrite or recomputation.

The primary point error, nonlinear event risk, classification, numerical gates,
sampling, losses, learning rates, seeds, adaptation counts and selected models
are unchanged. This is an output and recovery repair, with targeted synthetic
component and confirmation-cache checks. The new source identity is recorded
externally after application; C provenance is not rewritten.

To complete the approved discrimination role of new train data, after all 24
new train conditions have been attempted and locked, score the two frozen C
predictors main2701 and alternative2701 before any D adaptation. Use the existing
evaluation procedure, retain all condition failures and exclusions, and label
the result exploratory new-train discrimination of whole observable predictors.
It is neither independent confirmation nor evidence for isolated mechanism
necessity. There is no new model, condition, search branch, fit update, confidence
interval or primary-comparison selection. The fixed main2701 versus fixed2701
new-val comparison remains locked. This step has a finite 1800-process-second
cap inside the existing D allocation of 43200 process seconds.

All model and cohort identities and the wrapper hash must be registered before
that scoring execution. Any D adaptation evidence or confirmation freeze makes
the pre-adaptation scorer refuse execution. D then performs the original uniform
256 updates, freezes every formula, coefficient and initializer, and only then
generates and evaluates the 24 locked new val conditions. Original test stays
closed. All outcomes remain under the original scientific claim standards.
