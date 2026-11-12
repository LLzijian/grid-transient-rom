# 静态拓扑、隐幅值调制与力记忆：当前证据和识别边界

追溯编号：`gftd_kuramoto_discovery_20261106`。本页的当前综合更新至2026-11-12，直接依据闭合结论、审查与决定，以及当前两路拟合的终态、审查和部署释放；不新增实验或改变采用决定。整理前的[完整规划与状态原件](../reports/history/gftd_campaign/originals/gftd_kuramoto_discovery_before_synthesis_20261109.md)逐字节保存，原始预测与释放范围以原协议为准。

**当前研究候选是GFTD 5N、十机共50维自治ODE；固定共享参数的有界核同核初始化已覆盖原train fit297全部样本，匹配适配的两路fit已审查闭合，原全量部署已释放。有限K4静态恢复已有支持，隐调制恢复和真实预测收益仍待各自证据。模型没有晋升，`default=none`。原Run 3总体L1保留null，当前Run在同一train范围比较匹配适配后的两种完整方法。**

2026-11-09的[初始化精度Run](../reports/gftd_initialization_precision_20261109/conclusion.md)在固定共享参数、实际步长和门槛下完成全体594项验证：固定八阶初始化通过静态297/297、调制282/297，原47个输出全部保留。长时主细完成数分别为297/297、270/297；完成与长时精度分别判断。该Run未读取真实未来，原总体L1不回填，模型不晋升。

[长时数值辨别](../reports/gftd_long_horizon_diagnostic_20261109/conclusion.md)进一步区分了误差来源：五个保存的调制状态同时具有离散误差与三参考一致区间内的幅值增长；static_027的15 s参考精度仍未决。零项求值候选的20对真实检查没有恢复有限计算的案例，冻结总门未通过，条件轨迹未运行。随后的[储能可行性Run](../reports/gftd_storage_feasibility_20261109/conclusion.md)给出原族有限时间逃逸反例与有界核的全局线性增长证明；268两个旧状态的六条新参考全部1 s合格，独立复算通过。

[同核初始化Run](../reports/gftd_bounded_initialization_20261109/conclusion.md)已将有界核贯穿合法初始化：原297个train样本全部通过六项数值检查，289个完整1 s严格参考确认，8个参考到限保留未决；相对旧指数新增15个通过、没有丢失。当前[匹配预测Run](gftd_matched_prediction_20261109/protocol.md)已完成44项固定资格，两个模型的拟合已[完成并获审查接受](../results/gftd_matched_prediction_20261109/fit_adoption.json)，两模型各297包的[原部署阶段已释放](../results/gftd_matched_prediction_20261109/deploy_release.json)。最终模型与全部初始化选择冻结后，才读取真实train未来作同期L1，并报告同一最终ODE的L2/L3。

当前两路fit已[审查闭合](../results/gftd_matched_prediction_20261109/fit_adoption.json)：各四候选、合计192套数值检查通过，各三次共享提案分别得到同一参数，均未改善24包自治历史目标，最终保留各自父模型。静态45列和调制90列条件信息全部合格，联合识别仍unknown。最终48点相对投影梯度小，步长一致门静态0/24、调制1/24；调制15个起点到限，但各候选已选点均来自收敛任务。弱式条件提案与盒约束自治历史目标不同，具体影响尚未分解。详见[阶段科学判读](../reports/gftd_matched_prediction_20261109/fit_assessment.md)及[独立审查](../reports/gftd_matched_prediction_20261109/fit_review.md)；原全量部署采用4个worker，真实L1待全部选择冻结后评分。

## 当前认识：先按问题阅读

| 问题 | 当前判断与范围 | 原始来源与材料导航 |
|---|---|---|
| 什么信息条件支持静态边与权重恢复？ | 登记的四个 K4 无噪图均恢复正确支撑；含噪支撑仍正确，高噪最大边权误差约 0.264；平衡设置返回 null。此结论不扩张为任意图或 K10 保证。 | [原结论](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md)、[独立审查](../reports/history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_result_review.md)、[数值摘要](../results/gftd_sparse_topology_20261107/summary.json)、[Run 导航](../reports/gftd_sparse_topology_20261107/README.md) |
| 预测通过能否证明隐幅值调制已恢复？ | 人工条件中 48/48 配对数值合格、30/48 达到预测门；无噪调制条件中 beta=0 仍可通过预测门。机制采用门未通过，联合优化充分性 unknown；保留静态替代、信息不足与欠优化解释。 | [原结论](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md)、[独立审查](../reports/gftd_modulation_20261107/originals/gftd_modulation_20261107_result_review.md)、[数值摘要](../results/gftd_modulation_20261107/summary.json)、[Run 导航](../reports/gftd_modulation_20261107/README.md) |
| 原Run 3真实暂态同期比较说明了什么？ | train fit297 全部 297 包均进入流程，静态 46/297、调制 1/297 有数值合格输出；两模型对 30 个失稳样本均无合格输出。原总体L1指标为null，不能比较不同覆盖样本的均值，也不能把未覆盖写作零漏检。 | [原结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[独立审查](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)、[证据摘要](../results/gftd_fit297_20261107/evidence_digest.json)、[Run 导航](../reports/gftd_fit297_20261107/README.md) |
| 初始化检查的大量失败能否修复？ | 固定DP8区分并修正了积分精度问题；有界核同核初始化进一步取得原297/297六门覆盖，其中289完整严格参考确认、8到限未决。核改变并重新推断初态的结果与同初态纯核干预分别解释；真实预测尚待当前Run。 | [精度结论](../reports/gftd_initialization_precision_20261109/conclusion.md)、[有界核结论](../reports/gftd_bounded_initialization_20261109/conclusion.md)、[独立审查](../reports/gftd_bounded_initialization_20261109/result_review.md)、[root决定](../results/gftd_bounded_initialization_20261109/root_adoption.json) |
| 局部证书与物理对应分别支持什么？ | 两个冻结 ROM 均有非空局部吸引集证明，47 个可检查实际初态均未获覆盖；每模型为 10/13 模态配对，图相关使用 K(a=0) 与绝对 Kron 导纳。ROM 证书不等于全阶安全，球外不等于失稳，物理对应不等于真线路恢复。 | [结论中的 L2／L3](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)、[解释更正依据](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)、[root 决定](../results/gftd_fit297_20261107/root_adoption.json) |

### 认识修订与事件关系

- **真实比较的缺口已变化**：原“没有同期真实系统证据”更新为“同期流程已完成，总体效应因数值覆盖不足未决”，依据 fit297 的结论、独立审查和原分母结果。
- **L3 的局部解释更正**：[科学判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)中“参考角度有效刚度”的措辞由[独立审查](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)澄清为 K(a=0) 对绝对 Kron 导纳；[最终结论](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md)采用更正，相关数值不变。其余分析按原范围保留。
- **技术续跑不增加科学重复**：fit297 原执行保存 41 项后发生 I/O 故障，[续跑回执](../results/gftd_fit297_20261107/resume01/execution.json)继承这 41 项、完成余下 553 项，没有重拟合或重放；仍是同一 Run 的一次比较。
- **采用决定与归档完成属于两个时点**：[root 决定](../results/gftd_fit297_20261107/root_adoption.json)保留当时待归档动作，随后[归档回执](../results/gftd_fit297_20261107/archive_verified.json)记录推送核验完成。不回写旧决定来消除时间差。

### 原规划与当前综合的边界

下面保留原假说、架构和 Run 谱系；竞争解释表中的证据与缺口按上列原始来源更新。尚未释放的旧规划不构成新执行授权，科学状态、执行状态与工作量状态分别阅读。


- **登记性质**：`PROSPECTIVE_ARCHITECTURE_CAMPAIGN`（顶层设计规划，2026-11-06）
- **当前科学状态**。`active`。Run 1支持登记的四个无噪K4图恢复；Run 2接口闭环完成，调制恢复未获支持。原Run 3的总体L1因数值覆盖不足保持null；后续已区分积分误差与动力学增长，并完成有界核297/297同核初始化。当前两路fit已完成且均保留各自父参数，原全量部署已释放，后续仍须检验原分母L1及最终ODE的L2/L3。原模型的局部证书与10/13模态对应不自动迁移到新模型；代码与证据保留，模型不晋升。
- **授权引用**。用户 2026-11-06 确定 GFTD 核心方向，2026-11-07 明确恢复按评估方案研究推进。范围内正常推进无需逐轮用户确认，数据、部署与 TASK4 边界沿用 [kiki.md](../kiki.md)，真实状态以 [current_state](../docs/current_state.md) 为准。
- **范围明确的假说**。在给定观测与激励条件下，仅用合法因果信息，联合储能与弱形式结构选择能否区分静态耦合、动态调制和加性记忆，在可区分时选择简洁共享自治闭合，并改善 L1 预测。功率一致性、L2 证明和 L3 对应分别判断。
- **架构版本**。[v2.0 草案及更正](../designs/gftd_kuramoto_architecture_brief.md)；当前方程采用独立审查后的 [5N 修订](../designs/gftd_repair_design_20261107.md)，实施顺序采用[三步主线](../designs/gftd_mainline_20261107_architecture.md)。

---

## 1. 竞争解释与判别矩阵

| 竞争解释 | 区分预测（Discriminating Prediction） | 已有证据 / 反例 | 尚未解决的缺口 |
|---|---|---|---|
| **M0（动态调制改善闭合）** | 真值隔离后区分静态/调制边；固定初始化和共同模态算子时，调制开/关改变可观察预测 | 人工预测通过不足以证明调制恢复；有界核已有原297个样本的初始化数值覆盖，匹配拟合已闭合，条件信息合格但未接受共享更新 | 在原分母上评价两种完整方法的预测效应；分别报告条件结构信息与优化诊断，不能由预测收益单独证明调制识别或纯机制因果效应 |
| **M1（纯静态拓扑发现 $S^{\text{edge}}>0, S^{\text{mod}}=0$ + 线性记忆已充分）** | 无需将隐变量纳入耦合强度，仅靠稀疏静态藏本边与节点记忆即可在幅值跌落工况下达到同等失稳分类与轨迹精度 | 静态方案在登记的无噪调制人工条件下也通过预测门；真实静态与调制输出覆盖不同，不能由其均值证明静态充分或调制有益 | 保留最强静态替代；在相同合法输入与原评价分母下辨别，不用旧 21/30 漏检替代同期结果 |
| **M2（初始化/优化、共同参考或静态补救解释）** | 固定旧动力学，改合法初始化或给合理适配即可改变失配；最终两模型使用同一已核身份的物理参考 | 积分精度与原核增长已分别定位；当前静态192代理任务全收敛、96候选六门通过、45条件列合格，唯一共享提案未改善自治历史目标；最终24点优化充分性unknown | 区分弱式无盒记忆消元与有界自治轨迹目标，不能由条件信息合格推断共享目标下降；等待调制终态及真实预测，不追加当前拟合；部署释放时绑定原FOM/Kron缓存 |
| **M3（独立时变边权）** | 去掉联合反作用后，比较预测与功率变化 | 所选储能会出现额外边权变化项，其符号和实际效应须检查 | 按判别价值决定是否消融，耗散证明缺失不直接判长期发散 |
| **M4（噪声造成伪结构信息）** | 平衡下参数本不可辨，含噪字典却可能秩稳定并输出确定参数；信息强度规则须拒绝这些伪值 | 单尺度探针揭示秩稳定伪参数；后续静态拓扑 Run 的 24 个平衡设置均返回 null，含噪权重误差与支撑正确性分开报告 | 该有限对照不保证任意噪声或图规模；保留绝对信息与 null，不能只凭相对秩宣称可辨 |

---

## 2. 原规划与实际 Run 谱系（未释放项保留规划身份）

| 规划 Run ID | 父节点 / 起点 | 算法实例 | 类型 | 核心判别问题与验收标准 | 当前状态 |
|---|---|---|---|---|---|
| `gftd_repair_20261107` | v2.0 原型及父 RMemoryROM | 5N 独立 p/a/r | 数学与工程资格 | 43 项冻结检查通过；父映射、真实梯度、合法输入与存取，非零调制初始化和 fit 尚未实现 | `COMPLETED`，不晋升模型 |
| `gftd_observability_20261107` | 5N 修订及已完成资格检查 | 固定三节点局部观测探针 | `probe` | 8 基准、末点 p 硬约束、尺度化谱和所选方向未来响应，无拟合 | `COMPLETED`；非平衡调制组局部 7 秩、条件数约百万，保留证据 |
| `gftd_initialization_20261107` | 合格 5N 核心与局部观测探针 | 固定共享参数的末点 a/r 推断 | `probe` | 12 输入、36 优化；简单末点 a=0 达标 12/12，联合达标 8/12，充分性单列 | `COMPLETED`；简单方案为该人工范围工作基线，不晋升 |
| `gftd_endpoint_denoising_20261107` | `gftd_noise_confirmation_20261107` | 合法末点估计与同预算硬锚再拟合 | `discrimination` | 128 含噪输入×两方案；4 无噪复用；全分母保护与主细一致 | `COMPLETED`；候选 132/132、对照 131/132；同点纠正后全部采用条件获独立复核 |
| `gftd_endpoint_confirmation_20261107` | 已完成末点诊断及独立纠正附录 | simple600→smooth1200 完整流程 | `confirmation` | 64 个新方向、两档配对；所有输入从合法历史初始化，原预测门 | `COMPLETED`；14 项工程资格通过，新含噪 128/128、无噪 4/4，独立结果复核后限定采用工作初始化 |
| `gftd_stage0_unit_verification` | `bd4db25` | `GFTDKuramotoModel` v2.0 原型 | 工程与数学性质单测 | 2026-11-07 更正原“已完成”描述，六项单测通过仅支持对应条件，全部资格尚未通过 | `REPAIR_REQUIRED`，保留原测试结果 |
| `gftd_noise_confirmation_20261107` | `gftd_initialization_20261107` | 固定简单初始化 | `confirm` | 64 个新方向配对两档强度；低噪 64/64、高噪 63/64、无噪 4/4 达标 | `COMPLETED`；有限全通过陈述被推翻，保留基线与失败方向 |
| `gftd_stage1_kuramoto_recovery` | 5N 合格修订及初始化证据 | 部分观测结构推断，待设计 | `probe` / `discrimination` | 同族静态/调制正负校准，隔离图与隐轨迹真值，再检查独立生成机制 | `DRAFT_PENDING_RELEASE` |
| `gftd_coupling_memory_20261107` | 完整初始化确认与单共享尺度设计 | 关闭调制、固定边形状；弱形式解析消去记忆初态 | `discrimination` | 端口资格后，以同 11 点比较多测试与单测试；9 轨迹、27 观测包、264 次代数求解 | `COMPLETED`；无噪 c 误差 3.63e-5/4.62e-5，单测试 66/66 不可辨；含噪平衡存在秩稳定伪参数，独立复核后保留证据 |
| `gftd_sparse_topology_20261107` | `0cf535b` 单共享尺度证据 | K4 未知六边、完整端口和记忆消元 | `discrimination` | 四个隐藏图，三事件共享；无噪精确支撑/权重及零图、平衡对照；含噪逐项报告 | `COMPLETED`；四图无噪恢复通过，含噪支撑均正确但权重误差最大0.264；24平衡设置null，独立复核后保留证据 |
| `gftd_modulation_20261107` | Run 1 已归档3fd58f4 | 候选自洽幅值、精确指数列、层级门及 fit/initialize | `adaptation` | 12条件、24观测包、16模型和48完整初始化；因果接口、结构与预测分开判断 | `COMPLETED`；100项联合工程检查及17项续跑检查，48数值过/30预测过，调制恢复未获支持；独立复核后保留证据 |
| `gftd_fit297_20261107` | Run 2已归档66f1308；机制未识别 | 全体静态弱拟合、固定24包适配、全297评价 | `discrimination` | L1原分母、同ODE严格L2、既有descriptor与PQZ Kron L3 | `COMPLETED`；2fit、594初始化、47双步长预测。总体L1为null；L2点覆盖0，L3各10/13模态配对；一次无重放I/O续跑，工作量关闭且不转移 |
| `gftd_initialization_precision_20261109` | 已冻结的Run 3两模型与594份初态 | 参数固定，积分精度与初始化流程分开辨别 | `discrimination + adaptation` | [协议](gftd_initialization_precision_20261109/protocol.md)：同初态RK4与参考比较，依据结果选择一个修正并保留297分母 | `COMPLETED`；49状态诊断、固定DP8小试、594完整初始化和579长时配对闭合；静态297/297、调制282/297初始化通过，新增参考520/532确认；独立复核后保留数值覆盖结论、模型不晋升 |
| `gftd_long_horizon_diagnostic_20261109` | 初始化精度Run已保存的新初态及缓存 | 同参数、同初态的长时跨方法辨别 | `discrimination` | [协议](gftd_long_horizon_diagnostic_20261109/protocol.md)：10状态区分积分误差、接受轨迹增长与零项求值异常；只允许一个条件分支 | `COMPLETED`；46条积分均有回执，五状态在参考一致区间内增长且网格误差下降；零项20对真实检查19项通过、1项相同平方溢出未过冻结分类门，28条条件轨迹未释放，工作量关闭 |
| `gftd_storage_feasibility_20261109` | 长时诊断冻结模型、268两个旧状态及三参考缓存 | 同储能光滑有界调制，保留零点二阶展开 | `probe` | [协议](gftd_storage_feasibility_20261109/protocol.md)：先数学与人工资格，再两个保存初态各三参考1 s | `COMPLETED`；数学及人工资格通过，六条新参考全1 s合格；同窗幅值与调制范围改善，独立复核后保留证据，工作量关闭 |
| `gftd_bounded_initialization_20261109` | 储能可行性已归档989fa944，固定原调制共享参数 | 显式有界核贯穿字典、存取与合法DP8初始化 | `adaptation + probe` | [协议](gftd_bounded_initialization_20261109/protocol.md)：25包小试后完成其余272包，每个返回点一次严格参考 | `COMPLETED`；297/297六门通过，289完整严格参考确认，8到限未决；独立审查接受、源码归档，工作量closed |
| `gftd_matched_prediction_20261109` | 已完成的同核初始化及冻结静态／有界两父模型 | 两模型24包匹配适配、原297预测及同一ODE分析 | `adaptation + matched comparison` | [协议](gftd_matched_prediction_20261109/protocol.md)：固定两起点1200残差、完整库提案、原分母L1和L2/L3 | 公开接口和44项固定资格通过；两路fit完成并审查接受、均保留父参数；原全量部署已释放，真实评分等待594个初始化选择冻结 |
| `gftd_stage2_coi_decoupling` | 受控证据与合格修订，具体父候选待冻结 | 静态耦合候选 | `discrimination` | 在 `train fit297` 固定同一初始化和静态图，仅改变共同模态算子 | 旧规划保留；不再作为三步主线前置 |
| `gftd_stage2_latent_modulated_topology` | `gftd_stage2_coi_decoupling` 的冻结候选 | 动态调制候选 | `discrimination` / `adaptation` | 固定初始化与共同模态设置，调制开/关同期比较，L2 另证，L3 report-only | `DRAFT_PENDING_RELEASE` |

上述是规划引用，真实 Run/Job 身份、有限工作量和父候选在前瞻协议中冻结。数学反例和便宜辨别可直接推进相关问题。

---

## 3. 已用适配与信息暴露谱系

- 继承自 `force_memory_closure_20261030` 及前序 `retrospective_force_memory_20261030` 的 `train fit297` 暴露记录完整保留，不因架构升级为 v2.0 而清零历史暴露。
- 原始 `val/test` 保持关闭；旧已消费 `test` 永久退出模型选择；`new FOM = 0`。
- `train fit297` 的重复比较属于条件内辨别，种子重复衡量优化随机性。历史 `memory_transport_20261029` A/S/P 各组在真实可用区间漏检均为 21/30 个失稳样本、8 s 均为 4/30、误报均为 0/267 个稳定样本，该结果仅作动机，不能替代同期对照。

## 4. 下一个最有判别力的问题与重开/停止条件

- **当前工作**。[同核初始化结果](../reports/gftd_bounded_initialization_20261109/conclusion.md)已完整闭合：原297个样本全部通过六门，289个完整严格参考确认，15个旧指数失败新增通过。[匹配预测Run](gftd_matched_prediction_20261109/protocol.md)的两路fit已审查闭合、原全量部署已释放；接下来按冻结阶段推进最终模型、全297初始化与预测、真实L1和同一ODE的L2/L3。两模型对弱式提案与自治目标的判读已记录，当前科学协议保持冻结；数值覆盖不替代真实预测和调制识别。

- **2026-11-08证据更新**。M0/M1的同期流程已执行，但总体预测比较因数值资格覆盖不足保持未决；不再将“未执行真实比较”作为当前缺口。M2需要区分共享适配351/384任务到限与初始化后的数值资格损失：646初始化优化任务均收敛，却有547项未返回合格初态。条件支撑为静态35边、调制35边及25调制边，联合唯一性unknown。L2非空局部证明不覆盖本轮47实点；L3的K–绝对Kron导纳相关0.440/0.483及10/13模态配对用于描述，不替代L1。详见[Scientist判读](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_scientific_assessment.md)、[独立结果审查](../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_result_review.md)及[root采用](../results/gftd_fit297_20261107/root_adoption.json)。
- **判别顺序**。同族正负校准分开检查恢复和误检，推断端隔离真实图及隐轨迹，再比较独立生成机制与最强替代解释，随后准备 IEEE39 条件内辨别。
- **停止/转向条件**。无效实现先有界修复；信息不足或机制不可区分时检查可观测性和等价类，按证据决定修订或暂存。正、负、混合结果按五轴判读，确认需求与晋升资格分开。每个 Run 预登记比较、适配上限、停止/重开条件和实际成本，旧余额不转移。

当前完整L1闭合后，才决定是否需要[有限缩步草案](../designs/gftd_objective_alignment_20261112_draft.md)。[独立审查](../reports/gftd_matched_prediction_20261109/next_design_review.md)要求补齐逐次评价身份、实际有效系数和固定初态诊断口径；[root决定](../results/gftd_matched_prediction_20261109/next_design_decision.json)为暂缓采纳、不释放新试验。去掉确定性重复属于工程修正，三点缩步至多检验这次弱式方向，不能证明稀疏恢复或模型族闭合极限。
