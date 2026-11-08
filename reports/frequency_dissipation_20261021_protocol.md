# W8转速幅值耗散：有限同源研究

2026-10-21。root采用实际gpt-6-astra代理/root/astra6_after_edge_memory完整原案（reports/astra6_after_edge_memory_20261021.md，原始远端fe364ab44fae2b0a925f192528e730c33b91ca31）与独立挑战条件。原案的比较、检查、预算、评价、门槛和停止规则全部为本协议组成部分。E/P8/D保持关闭与源冻结，不进行修复或重试；本轮不是其补做。

## 唯一科学问题与定义

检验物理转速幅值相关的额外耗散，是否改善共同频率而不压制真实失步。H_W=H_N7，u_W=u_N7+b_W*(omega/input_scale_omega)^3，b_W<=0直接精确0初值。AST严格port/pow3(var w)，共享一个系数按term_key定位，旧scale必须严格正且不改。无q³端口/q⁴储能/新增边储能。固定旧参数新增端口原点值/Jacobian均零；联合训练后局部谱重新算，不宣称保持稳定。

b_W属于原closure组；Adam后检查参数/梯度/矩有限，只对b_W做min(0)投影，不重置矩。新增dotomega=b_W*(omega/s)^3/M，经原动量B方向；外部u完整替换自治端口时连同W项替换。Hpi=omega与B/单位先核，新增H率b_W*sum(omega^4/s^3)<=0仅是这一端口的耗散恒等式，不是完整H单调/全局稳定/吸引域/FOM证书，单位H-unit/s不是MW。

## 完整适配和复用

只新增6101/6102/6103三个W1024。每seed原O512同源、新Adam，复用quartic_memory01/prepare01冻结1024日程、原RNG/coverage。CUDA float64 RK4 dt=.01，原1/2/5s各342/341/341、10随机+2覆盖、force1.0068379869405584、辅助采样/尺度与原目标10risk+local+.1hidden+.1torque、clip100。LR .0009/.00009/.000027，第513起乘.3不重置矩；stage0与每32更新保存33份，1024固定端点，512只保存。全部旧骨架/闭合及一次GRU联合学习。主seed6101不重选。

原O512 manifest b8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4；N/Q冻结manifest a1b411dae2c00867611afe390d934b8cc718d55d3103dd81aba0b6229aa34842、完整指标4f53916743184e269541f42797a829bbcf938b7a8047b18978147f7d9f107820。主W−N，完整W−Q；旧N/Q不重训/重评。新消费者必须实际核三源公共状态/GRU/flags/场、日程、RNG/coverage、空Adam及组/LR/513衰减、目标/cohort/scale/force/采样/辅助位置合同。E未完成的合同不称通过。

40状态、100ms一次因果GRU、真实单位/标签/区间及数值门槛保持。train385=原366+新19=342稳定43失稳；原27排除/新5失败保留，不补尾/删除难事件。val/test关闭、TASK4暂停。P1搜索关闭，18×512的6085.6472446不重跑；P3额度用完及旧24val guard失败不重试/不搬模型，P4同候选门槛不绕。

## 工程、必要检查与入口

新experiments/frequency_dissipation_runtime_20261021.py、frequency_dissipation_20261021.py及三类专用测试，BASE=private_data/closure_mechanism_20261019/P2/frequency_dissipation01，variant W。专用WRuntime/build_W/enable_runtime/coefficient_indices/one_update/extra_port_power接口；全自动微分图内映射，encode/field/rollout/export/from_export兼容；旧代码不改，不新增性能平台/prepare/科学日程。复用P0冻结合成资产（manifest0a1f2463693b0a5227c8f39e725a4e8891af0300dfccdfde705d66e346ee9816）。

独立实施审查后唯一QA01：合法八AST、非重复、零点值/Jacobian、正scale/Hpi/B，b0与N场/GRU/公共raw梯度、新b连通；generic/映射/NumPy/非单位M/外部u、共同角与机器重编号；新增H率非正；完整5s原目标/全raw梯度/clip/Adam/投影与保矩、非空文件恢复/CPU-CUDA-NumPy RNG/参数唯一归属/513LR；因果/真实n=1/不等长/非train/来源/重复输出/cap拒绝。field rtol2e-11 atol1e-11、loss2e-10、grad/Adam2e-9不放宽。

新增独立子进程从实际cwd、PYTHONPATH未设运行干净python -m生产入口：必须实际走入并成功完成合成baseline合同入口与必要模块解析，在真实measure_cost前退出，不只--help，不靠pytest预插入sys.path。QA证据说明合成入口覆盖范围，真实旧N/Q合同仍须cost实际核过。新runner须在任何依赖调用前建立自身包路径或明确包导入；禁止修改旧E路径来营救。

QA01一个可定位纯工程故障才允许修复后QA02，总帽不变；不改科学定义/样本/mask/断言/容差/范围。QA01通过禁止QA02。真实cost仅一次、失败无修复重跑；训练/eval无重试/外部恢复/延长。每次实际启动前归档并核命令/prior/binding/source，核真实PID/精确整数FILETIME/退出/GONE/锁，禁止重复队列。可复用E失败receipt中344来源前后相同的身份基表，不称该cost成功，完整native before/after重hash不可减少，不整库hash混合HDF5。

## 9500进程秒硬帽

P2总9500：support hold800（含已计planning600，仅新增200），静态实现/绑定/准备180（初分runtime50/runner70/root60），QA最多2×native200/PS180/self160=400；cost500/470/440；3train各2200/2170/2140=6600；3eval各240/210/180=720；冻结/提取/费用元数据300，原生监督个体<=100/70/40。子进程只计一次，失败/拒启/重试/监控均累计，不挪类别，0<self<PS<Max<=3600。

E关闭后P2余9510.309738700118，规划已扣600后8910.309738700118；本次仅新增预占8900，余10.309738700118不延长本轮。父规划后累计128423.05456179997，新增support200后累计128623.05456179997/余130576.94543820003；尚未执行最大8700另保留，父未分配121876.94543820003。P0/P1/P3/P4/repair不互借，旧900和P8未知尾hold保持。

真实cost固定12最长/12最短失稳两面板×1/2/5s×2完整更新=12，丢弃状态；同步全cache/forward/backward/clip/Adam/投影，各h取最慢。三O512零更新等价/实际加载、非空Adam1→5恢复见证占已有更新。准入1.25*(342*t1+341*t2+341*t5+33*t_save+t_load+t_extra+t_supervision)<=2140/模型，3合<=6420；save内export不重计，额外未发生export为0，监督20秒beforemargin另列。固定最长16真实事件CPU double单线程dt=.01/.005、完整15s/评分/压缩/39维导出和本轮摘要，实测×25×1.25<=180/eval。已花与剩余最大帽都须<=9500。未过即关闭W计算分支，不重测/涨帽/少seed/少步/减评价/拆恢复；合成加速不放行。

## 评价、门槛和关闭

三有效1024端点先统一冻结归档，再各一次完整385双步长评价，缺角色不称完整比较。共同385/原366/新19分开：稳定1/2/5/8s原风险/COI/共同频率rad/s/Pm100MVA pu及MW；43失稳真实区间至15s原点平方误差/共同频率及功率RMSE；8s与真实区间分别漏检/误报/未决/触发时间。原数值资格/分母不变，失败样本保留、相关总体null，不15s救8s或只报合格子集。

只消费同次预测计算P_W=b_W*sum(omega^4/s^3)：真实时间梯形积分/均值/最小值；新增转速场范数均值/均方/最大，单位rad/s²（接口转速每秒）。t0一次+n未来点，n=1一个真实间隔；非有限摘要置对应null，不新rollout/重编码/关项反事实。记录b raw/裁后梯度、投影前后/Adam增量/投影次数、clip及各h前后64次原loss，持续改善/投影按实际限定优化，不称收敛/机制不存在。

原候选门槛全精度不加容差：三seed完整数值合格，各seed稳定5s风险/共同频率RMSE及失稳真实区间三误差均不增，两范围漏检/稳定误报均不增；至少2seed含6101真实区间漏检严格减。W−Q不替代W−N。至多一个后续研究候选，重大采用前独立挑战，默认/确认/P4不自动改变。未过关闭具体结构并留所有收益/损害，不立即扫符号/阶次/尺度/邻式。

独立挑战及root决定先登记后执行，合格工程经限定审查PR/expectedhead/远端验证，模型按科学门槛另决。不合格不默认采用；证据/源checkpoint/负结果保留归档，不reset用户工作/强推/删除。每一结局选择性GitHub归档。本轮关闭后按实际余额核全部合法类别并实际推进下一独立问题；若无可辩护完整分支才记真实全局阻塞，不能仅凭W成本失败停止父任务。
