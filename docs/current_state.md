# 当前研究状态

**训练未恢复；`closure_probe_20261030` 已完成（probe，optimizer 调用 0）。** 用户 2026-10-30 指令“按项目规划开始推进”后，本轮只做了已授权数据上的无训练分析：未开 val/test、未新增 FOM、未训练、未晋升模型；TASK4 暂停、default none 不变。执行状态与训练授权状态分开记录，本轮不扩大训练授权。

**本轮结论。** 冻结 train fit297（已消费，非独立确认）上：H2 初始隐力可辨识性**不支持**为主要解释（r₀ 设计 cond 1.613、297 条事件逐事件相同，ridge 对 σmin 贡献 9.9e−6，ridge 扫描 r₀ 变化 median 2.7e−5；A/S/P 适配族内 r₀ 差异 ~1%，A96 4.5%）；H1 共同模态闭合**降级、未否证**（COI 预算中 Δr 占 75–77%，但标量闭合残差主要跟随 W1 轨迹差异，放大倍数 1.02–1.09、W1 差异解释 64–71%；A−R_N 稳定差距不在共同项）；**H3 优化/选择强支持**（全队列 128 比 64 差 5 s +0.050289、8 s +0.026602；九点曲线非单调，5/8 s 最小值都在 step16：0.752083/0.576173，两项都优于 R_N 0.830421/0.591762）；H4 机械体/端口**开放**（D、q_p 非均匀；p 通道约 1/3 误差来自固定 p 动力学，探索性、非预登记）。step16/64 只是事后中间 checkpoint，**不晋升、不选为模型**，也不改写已冻结的 128 结论。

**证据与指针。** 结论 [closure_probe_20261030_conclusion.md](../reports/closure_probe_20261030_conclusion.md)、协议 [closure_probe_20261030_protocol.md](../reports/closure_probe_20261030_protocol.md)、简报 v2 [architecture_brief_20261030.md](../reports/architecture_brief_20261030.md)、挑战 [architecture_brief_20261030_challenge.md](../reports/architecture_brief_20261030_challenge.md)、Campaign [force_memory_closure_20261030.md](../campaigns/force_memory_closure_20261030.md)；公开包 `results/closure_probe_20261030/`、`results/coi_budget_20261030/public_budget.json`；harness 记录 `harness/runs/closure_probe_20261030.json`（assess valid=true，promotion_eligible=false）。

**成本。** P1 23.6 s、P2 2.8 s、P3 52.7 s、P4 111.3 s、P4b 408.7 s；CPU float64、单进程；optimizer 0、模型 rollout 0（P4/P4b 为前向评估，不是训练）。旧 `memory_transport_20261029` 的 492 次调用与 7472.72 s wall 是上轮范围，本轮不混用；`memory_transport` 未用 1356 仍关闭、不转入下一轮。

**上轮与档案。** `memory_transport_20261029` 维持 COMPLETED / NOT_RETAINED（A/S/P 各 128，无延长、无第二种子）；R_N 保护档案、A96 来源、S/P 端点、失败与恢复证据原位置保全；旧 `r_memory` 仍为 INTERRUPTED_RUNTIME，1165 实际/1152 持久，正式总成本 UNKNOWN，不得用本轮成本代替全项目账目。

**下一动作（均未启动）。** (1) 首选：登记一个**有限、预登记的优化/选择 Run**，在 key events（失稳漏判、误报、真实区间风险、时机）上评价 checkpoint 曲线并写死选择规则与确认方式；属 adaptation/discrimination，需新的 Astra 有界设计，不能沿用 E/s 余额。(2) 若追 L1 精度：先做 **p 端口预登记诊断**（固定 p 动力学 vs 最小扩展）。(3) E/s 结构扩展只在出现不依赖机械反演混淆的闭合不足证据后再考虑。以上都需用户确认或新的 Astra 设计；本轮不自动开始。

**代码决定：** probe 实现、测试、报告与公开结果选择性提交到研究分支，未改 main 科学模型。**模型决定：** no_change（default none，step16/64 仅证据）。**科学决定：** 共同模态闭合问题 unresolved；H2 不作主要解释；H3 为优化/选择贡献因素；H4 开放。
