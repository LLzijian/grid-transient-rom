# 检查点选择审计：独立只读审查

**结论：冻结设计可执行；所审评价实现未发现必须阻止有限无训练评价的问题。** 本轮可登记 11 个评价对象（A_30311 的 9 个检查点与 A96、R_N 两个参照），其中 3 个对象复用原主细证据，其余最多 8 个新增完整主细评价。最终建议起点须另经冻结的身份、固定时机集合和来源健康审查；当前不作模型晋升或候选通过判断。

日期：2026-10-30。task_id：`/root/review_checkpoint_audit`。role：Reviewer，与设计及实现作者分离。configured_model：`gpt-6.1-sol`（root 交接任务配置）；server resolved model：`unknown`，本任务工具未提供该字段。此报告帮助 root 判断是否启动有限评价，以及最终建议起点前还需核验哪些证据。审查为源码与冻结文本静态核查，未读取任何私有事件、模型、预测、Adam/RNG 载荷，未运行真实评价、训练或 optimizer；没有把实现者报告的测试通过改称本审查的独立实测。

## 1. 实际读取的冻结输入

已读取 AGENTS.md、kiki.md、docs/research_policy.md、agents/runtime.md，并按 anti-defensive-writing 的适用报告原则检查结论范围。

| 输入 | 实际 SHA256 |
|---|---|
| reports/checkpoint_selection_20261030_protocol.md | C2C1325229377BED6D0151A45891020A71C376B91F91BC1A71A5320134F76FFB |
| reports/checkpoint_selection_20261030_scientist.md | E3E22F83EA6587CA6F574CE351CC113E0CF5D5F8DC2275BD1F89F5B17C753B27 |
| experiments/checkpoint_selection_20261030.py | ED9D89C98A907E0EE3972FB41CC334F252DAC9DA31B544F82B819AFB22E15957 |
| tests/test_checkpoint_selection_20261030.py | 8EA61A875E7A5C0962B43090C846856026D9D1EBEFF1BE2414C25D24847CC058 |

Scientist 哈希与交接指定值一致。评价实现仅在 root 通知冻结并给出哈希后读取；未审查其写入过程。此审查不包含 root 后续编写的纯决策分析器。

公开原合同源码也已实际读取并取哈希：

| 原合同源码 | SHA256 |
|---|---|
| experiments/memory_transport_study_20261029.py | 1CB35AE27317A3ECF4B852895E6EB8564E5517D9B9CB75E14231DCFDE16E252F |
| experiments/state_gated_memory_study_20261028.py | DCB5D2CAC1462D4F71241E888EE77DB34A143F2F7A80091A34B249E18E6967B9 |
| experiments/fixed_edge_pair_20261019.py | 6F3194387F13A103A04BD73D27DBE17CDE7E786D8617655BA2FFF8BF811D72C3 |
| experiments/frequency_dissipation_20261021.py | F83EDEEDF24C40078DD92E2BDAAF3AF9739B838027B4179167341218D6188C7F |
| experiments/evaluate_structured_rom.py | 38AACBF79E61081449ADCB254D99F0556F65021D2BD5B8D46CF17C5D757E01D9 |
| experiments/active_fom_pair_fit_20261020.py | BFC74E84B8ACED9588618B60DD6440ADCBF96C1DC6610EA60001425EEC623720 |

另只读核查了 closure_probe_20261030.py 的冻结数据加载和模型加载函数、force_graph_study_20261028.py 的 profile 资格，以及 force_exchange_study_20261028.py 的指标汇总；未把这些只读行为解释为私有证据核验。

## 2. 设计审查

**原 efficiency_target 全部门保留。** 对照 memory_transport_study_20261029.py:241，设计保留候选、A96、R_N 三者 `sgm.valid`，稳定 5/8 s risk 与 8 s common ω/normalized p RMSE 各不超过 R_N 的 105%，真实区间失稳 risk 不超过 R_N，真实区间漏判不超过 21/30、8 s 漏判不超过 3/30、两时域误报为 0/267；实际改善仍为真实区间漏判不超过 20/30，或 risk 不超过 0.95×A96 且风险差严格大于双方粗细扰动之和。risk 与 RMSE 分开命名，没有以其他误差替代原门。

**数值合同不变。** 设计引用原 scientific_qualification、actual15_qualified 和 sgm.valid；真实区间 `n=min(1500,len(x)-11)` 与 fine 抽样/真值索引相符。原资格要求相关全部细子步与粗步有限、COI/common 归一化 RMSE ≤0.001、粗细分类一致及共同检出确认时刻差 ≤0.02 s；不以删事件或改步长消除未决。0.01×(confirmation_index−10) 沿用初始化末点时间原点。

**身份保护具有明确用途。** 每个时域分别保护 A96 与 R_N 合格检出集合的并集，防止平均收益或漏判总数不变掩盖错误身份替换。该守卫只约束本次工作起点准入，不能追溯改判旧 Run，也不能把因该新增守卫未通过解释为原 efficiency_target 失败。

**T 的分母已被正确固定。** 原 timing_comparison 的候选交集可能缩小；新设计先由 A96/R_N 共同可定位事件固定 T_anchor，要求候选全部覆盖，核对 truth 时刻一致后在同一集合比较 E/u。空集合、缺事件、缺确认时刻均不能通过。候选额外检出单列，不能用于改主比较分母。该新增保护与身份守卫是有具体风险对应的本轮前瞻规则；未见额外增加结构、最少模态数、优化充分性或其他无关硬门。

**信息暴露和确认表述适当。** 稳定 5/8 s 九点曲线已暴露，step16 的最小值是回顾性描述；同源 train 上新增关键事件读出仍是选择审计。未来两条训练日程只检查优化随机性，不能自动成为独立数据确认，也未授权当前执行。短于 64 步的来源标 insufficient_history，不能冒充通过旧 64 步诊断，也不应因此自动判优化失败。

## 3. 冻结评价实现审查

- `register` 固定 fit297 精确 ID、唯一性和 267/30 标签计数，核验 fit 缓存，绑定全部模型、协议与源码；完成前再核对来源、模型、数据和协议哈希。数据加载沿 checked_preparation 的 fit_only 路径；没有打开 val/test、新增 FOM 或 optimizer 的入口。
- 三个复用对象恰为 R_N、A96、step128；固定 evidence/model 哈希、完成回执、预测文件哈希、297 个精确事件与历史、主细预测形状及 history_dt 均检查。复用由原预测重算缺失摘要，并要求重建 profile 与原 state_gated_profile 完全相等，失配会停止，不会静默换参照重算。
- 新评价直接调用 sgm.evaluate。其 paired_evaluate_model 采用 CPU float64、单线程，未来 0.01/0.005 s，带 history_dt 的模型分别重新使用 0.0025/0.00125 s 初始化；参照 R_N 沿原无 history_dt 的加载与评价路径。
- 评价异常与非有限预测按原 helper 保留事件记录和未决指标；新实现保存原分类计数并核对身份投影计数一致，不筛掉数值未决事件。分类集合可以与未覆盖/判据未判定重叠，报告中已说明，不应把这些集合当互斥分区相加。
- 旧推理 seconds 保存在 original_inference_cost，当前新增审计耗时另存 current_audit_seconds，复用标记明确；不会由该字段直接产生同精度加速结论。完整新增总成本还需 root 汇总实际进程范围，不能累计旧推理秒数。
- 完成回执绑定 manifest、summary 和产物；已完成对象验证后跳过。缺终态回执的尝试拒绝重放；失败保留。`--retry-failed` 是操作入口，不自动授予重试额度：执行仍须遵守协议的一次有界技术修复，不得对真实数值门失败反复重试。

实现者报告 11 项人工测试已通过；本 Reviewer 已读取测试，覆盖导入无载荷、身份/历史、预测哈希、重复 ID、回执和重放保护等。未重复运行测试，也没有使用这些测试替代真实证据核验。

## 4. 执行与最终判读的分界

评价层可以启动，建议先完成 R_N/A96 的复用核验并封存 D*、T_anchor、truth 时刻一致性及哈希，再启动其余九个候选对象。这样可直接满足“先冻结锚、后读候选”的协议顺序；最大新增主细评价仍为 8 次。

所审评价源的 `timing_common` 和比较摘要是描述性候选交集，没有资格决策；`selection_rule` 和 `selected_checkpoint` 均为 null。这不阻碍存证评价，但不能直接拿该交集摘要判固定 T 门。最终分析器必须先检查 T_anchor 全覆盖，然后在相同 ID 集合上计算 E/u，且单列相对每个参照的新增漏判、修复漏判、新增误报与未决身份。

A0/A96 原双向兼容证据、来源 trace/Adam 有限性和未对账 optimizer 调用审计不在此评价实现内；这些证据缺失时可以保留评价结果，但相关来源资格保持未决。A128 的重建 profile 完全相等比登记的 ≤1e−9 指标复现容差更严格；若出现差异，应按技术证据失配处理，不能放宽门槛。

全部九点封存后再作完整门矩阵、分项非支配集和最早步数建议；不得提前依据 step16 或少数已完成点停止选择。当前三个建议决定为：代码 retain_branch（评价层）；模型 retain_evidence（未评结果、未晋升）；科学 continue 进行本轮有限审计，尚不宣布存在合格起点。default none、原 val/test 关闭、新 FOM=0、optimizer=0、TASK4 暂停保持。

此审查证明冻结文本和所读评价实现之间未见启动阻断；私有参照真实性、实际运行结果和未冻结决策分析器仍须各自的证据与后续审查。

## 5. 决策分析器追加审查（2026-10-30）

**结论：所审决策分析器正确实施已冻结的指标、身份与固定 T 选择规则，可以在 11 个对象完成后执行纯决策计算。其建议必须暂定，直至 root 完成来源健康与 A0 兼容资格审计。** 当前追加只读源码和人工样例，未读取正在生成的私有结果或预测，未执行决策器、真实评价、训练或 optimizer；不改变任何模型或代码指针。

本次追加前，本报告原版本 SHA256 为 `227265E9592C58A4D83182131A74F319E3AED0239C9AB511B3FA173C5BC2620E`，原评审内容保留。

| 新增冻结输入 | 实际 SHA256 |
|---|---|
| experiments/checkpoint_selection_decision_20261030.py | 4E3AB696ECE65F61F252AD35D8A7BDED74363251F9EC79A4FF3300282E0B9559 |
| tests/test_checkpoint_selection_decision_20261030.py | 59BC1294CA30645C2AEF1927BC84D6D648CA8C4D24F3E3B3A9784E49EB5A2DB2 |

两者均与 root 交接值匹配。人工样例 13 项通过由 root 报告，本 Reviewer 已读其测试及样例，但没有独立重跑。原 efficiency_target 源码再次取哈希，仍为本报告 §1 的 `1CB35AE27317A3ECF4B852895E6EB8564E5517D9B9CB75E14231DCFDE16E252F`。

### 5.1 已核对的选择逻辑

- 主入口 `load_completed` 必需读取 anchors.json，绑定评价 manifest 与 A96/R_N summary 哈希；11 个 complete 回执缺一即拒绝进入决策。逐对象继续核验原回执、summary、模型身份、per_event 产物、精确 297 个 ID/标签和 timing 与 profile 一致性。纯函数 `decide` 也要求完整 11 对象；其可不传 anchors 的分支供人工测试使用，正式 main 必传已加载的 anchors。
- D* 由两参照逐时域合格失稳检出并集重建；T_anchor 由参照共同完整且数值合格的 truth/fine/coarse 确认事件重建。锚文件的并集、排序 T 和 truth 时刻须逐项与重建结果精确相等；参照 truth 不一致即拒绝。不会凭锚文件中的声明跳过重建。
- 每个候选调用原 efficiency_target，另逐项展示原绝对门。新增身份守卫检查整个并集；T 缺事件、缺确认时刻或 truth 不一致均不能通过。E/u 直接用完整 anchor 计算，缺覆盖时 E/u 为 null，不能以缩小的 common_event_ids 计算门槛。
- 固定 T 上分别与 A96、R_N 比较 `E_c≤E_ref+0.01+u_c+u_ref`。额外可定位事件另列；八个原分项按实际数值逐项求非支配集，再选最早步数，没有合成权重或稳定风险回退。
- 结果保留完整失败门、逐参照错误身份变化、固定锚时机明细；公开副本删除事件 ID，保留计数与集合哈希。optimizer 当前判定为 not_applicable；`model_adopted=False`、`retain_evidence`、未来训练未执行等字段明确。

### 5.2 资格与解释中的两项必要限定

**来源资格尚未进入此分析器的门。** 第 199 行把来源优化健康写为 insufficient_history/unknown，明确 `audited=False`、`gate_applied=False`；第 205–206 行仍可输出 `eligible_checkpoints`、`suggested_adaptation_source` 与 `work_start_found`。因此这些字段只能解释为“评价指标、身份及时机门通过后的暂定建议”，不能解释为来源健康/A0 兼容也通过，更不能触发采用或训练。若 root 后续查到实际非有限状态或未对账 optimizer 调用，对应来源资格须保持未决；不能用本轮 optimizer=0 的 not_applicable 抵消来源问题。短历史本身仍不自动判失败。

**原门函数需要单独保全来源。** 评价 manifest 的附加源码列表没有显式包含 memory_transport_study_20261029.py；决策器会调用该文件的 efficiency_target，并在产出时记录其当前哈希。这不影响当前已冻结源码的数学审查，也不阻断正在执行的评价，但 root 应在决策执行前核对该源码仍为本报告的哈希，并把核对值和决策器哈希一起归档。锚文件本身的读取哈希也写入决策 inputs；“确实在候选结果读出前冻结”的时序证据由 root 的冻结记录提供，源码重建检查不能单独证明历史时序。

在以上限定下，未发现需要修改已冻门槛、身份集合、T 分母或 Pareto 规则的问题。此次结论只批准纯决策计算的技术一致性；最终科学解释及任何建议采用由 root 完成来源审计后分别记录。

## 6. 来源资格追加审查（2026-10-30）

**结论：新增来源审计足以在已存记录范围解除“实际来源尚未对账或已发现不健康”的混淆。step64/80/96/112/128 通过原 64 步健康检查；step0/16/32/48 保留 insufficient_history，已有有限性与来源对账证据不等于通过完整 64 步门。** A0/A96 的原双向 fit297 兼容及共享导出状态相等已有明确公开审计结果。此次补充不证明优化充分，也不提前判断尚未齐备的九点评价结果。

本次追加前本报告 SHA256 为 `77C4D6F9A319CB76228C5AA151FF2FE90A222507990CE0D280B29BFD6A9DFD91`；§1–§5 及其历史判断原样保留，此节更新来源证据的当前解释。

| 实际读取的新增输入 | 实际 SHA256 |
|---|---|
| experiments/checkpoint_source_health_20261030.py | 1DE3CF11769B652989C04D2336EA0576B657CA6DB9002D798645DD528C5827F3 |
| results/checkpoint_selection_20261030/source_health_public.json | 1EE7436456E85B332C519026A451D39B201A308E876B51395877E392C9A6F2BF |

脚本哈希与 root 指定值一致，公开结果记录的 source_sha256 与所读脚本一致，original_health_rule_sha256 与此前冻结的 memory_transport_study_20261029.py 一致。本 Reviewer 仅读取这份公开结果与源码，未打开私有 checkpoint、Adam、trace、journal、兼容记录或新增预测，也未执行此脚本或任何 rollout。

### 6.1 对账与健康检查的证据

脚本逐点先核对模型与 training 文件的元数据哈希，再检查导出模型数值、training tensor 和 Adam 有限性；核对 checkpoint/state/目录步数、完整 trace 步号、与 step128 trace 的相同前段、Adam 计数、分组布局、动量形状和原源码/A96 来源哈希。journal 对账检查 attempt/completed 对应、重复调用、未决调用和正式 1–128 步序列。公开摘要九点的 17 项检查全部为 true：项目 492 次 attempt/492 次 completed，其中 formal384、QA108，A_30311 正式 128 次；项目和该 Run 未决调用均为 0。

有完整 64 步窗口的五点原规则均为 `optimization_unresolved=false`，最近 64 步裁剪数与各块大变化数均为 0。短历史四点原规则为 null，没有把较短窗口的统计冒充完整 64 步检查；公开的所有已存梯度与更新统计仍可用于说明没有已记录的非有限或异常大更新。step0 的梯度统计为 null、Adam 计数为空，与零步含义一致，不伪造观察。

这些检查支持“已存资产与调用已对账，已检查部分未发现健康异常”；它们没有验证该实例已获得充分优化机会或全局最优，也不构成泛化或机制识别证据。短历史建议仍须把 insufficient_history 随建议来源一同保留，不因其长度自动否决，也不强行改标完整健康通过。

### 6.2 A0/A96 兼容与来源绑定

公开结果列出的旧兼容记录哈希匹配、双向标志、原 8 s/真实区间数值范围、精确 297 个 fit ID、297 行全部合格、A96 evidence 和零点完成回执均通过。共同导出 state 全部逐项相等，额外状态仅为固定平面 u/v，A0 variant=A、history_dt 均为 0.0025。相应状态为 `qualified_existing_bidirectional_evidence`，没有新 rollout；这支持原 A0/A96 数值兼容和共享状态一致，不能改写成两个不同模型类的字节导出完全相同。

本公开摘要删除了逐点 assets，保留私有证据的内容摘要。因此本独立审查没有直接重算私有模型/Adam 哈希，也没有把“源码正确且公开检查通过”写成亲自验证了私有 tensor。root 已掌握私有完整证据，应将本摘要字节哈希与评价 manifest 中同路径的九点模型哈希、A0/A96 模型哈希一起绑定归档；公开 `private_source_health_sha256` 是排序 JSON 内容摘要，不应误称私有文件字节哈希。

### 6.3 对先前暂定条件的更新

§5 所述来源未审计条件现由这份独立来源审计补充，不能继续笼统报告所有点 health unknown。最终 root 解释应叠加两层结果：评价器/决策器提供指标、身份与固定 T 门；来源审计对五个长历史点给原健康检查通过，对四个短历史点给已对账且保留 insufficient_history，并记录 A0/A96 兼容。原决策器中的静态 unknown/audited=False 字段不自动反映这份旁路审计，须在结果文字与输入绑定中明确关联，不能悄悄改写已封存产物。

公开审计新增 wall 为 2.886849 s，optimizer=0、训练时间=0、新 rollout=0、val/test 未读。各 checkpoint 的 recorded_source_training_seconds 是同一优化路径累计到该点的旧时间，不能在九点间相加为独立训练成本，也不计入本轮新增训练成本。当前仍不改变任何指针或启动训练，且不作优化充分性、模型晋升或候选指标通过的主张。

## 7. 最终结论与公开结果审查（2026-10-30）

**最终科学审查决定：接受“九个既有检查点均未同时通过冻结目标，保留证据、不推荐来源、不晋升；科学状态 unresolved”的结论。** 表格数值、step16/step96 的具体权衡、固定 T 分母以及三个决定与公开结果一致。结论稿有一处产物位置说明需修正：公开逐点 JSON 的 horizon_risks 只有 5/8 s，不能称其完整保存了 1/2/5/8 s 风险；该文字修正不影响选择结论，不需重算预测。

本次追加前本报告 SHA256 为 `9E71E4535DD3BA169A6EB6C88D1CE357716582A3F6E3274D8813480B0F939A39`。此次只读结论稿和公开摘要/回执，没有打开私有预测、重算指标、复跑训练或新建 Run；未展开机械 p 端口草案的机制审查。

### 7.1 输入及冻结一致性

| 实际读取的最终输入 | SHA256 |
|---|---|
| reports/checkpoint_selection_20261030_conclusion.md（本轮初读版） | E5076055C0E0FAC5388487ACB6CAEA2AEC50370AFD4FA76366ABE96790080C27 |
| results/checkpoint_selection_20261030/decision_public.json | DBA40315471D95BBB3AACAE9E18E7E0B2FA296A4A27A0AD2A3EECB9009AFC7C2 |
| results/checkpoint_selection_20261030/source_health_public.json | 1EE7436456E85B332C519026A451D39B201A308E876B51395877E392C9A6F2BF |
| results/checkpoint_selection_20261030/execution_receipt.json | 072C9D2FF968CB0B87C181683D39E8F63B47E94773BF32B011F55D3A734F7E92 |
| results/checkpoint_selection_20261030/anchors_public.json | E51D29C9412F5D4F14C581839FE4581B6BB43A38A76B543AF8E4F6968B8EB98E |
| results/checkpoint_selection_20261030/stage2_release.json | 37182089A59F53896476EA5F394E5C3B05BD710FEE78D46F48F3DD22328802F7 |
| results/checkpoint_selection_20261030/decision_release.json | F62947E6B9F575DAFD6E7CB834F81D4A4310AEBE9B470A7B9EA7965DB65BB40E |
| results/checkpoint_selection_20261030/execution_release.json | 1C01228807C855C7E24511085DEEA41A6D6AF0F97BB012DC87924D30240B7A88 |

再次取哈希确认评价器、决策器、来源审计器、原 efficiency_target、协议与 Scientist 均与 §1/§5/§6 的冻结值一致。decision_public 的 source 也记录了相同协议、Scientist、决策器和原门函数哈希，未见观测结果后改门。

十一份公开逐点 JSON 实际读得以下哈希；它们的 manifest_sha256 均为 `9413d62bab914500c2049506f01fd8a70998ac1c7f011c06f8574242a4bc0686`：

| 公开逐点文件（results/checkpoint_selection_20261030/） | SHA256 |
|---|---|
| R_N.json | 66021758BC12EE78A1950DBB99D42C9F1171B10158A89750CBD2739C78155007 |
| A96.json | E5C838D1F306FC0C687B8BAA43DC09742098002714C3DE56B7ACFA5FED1549C3 |
| step000000.json | AE26B4DED2C434D2BD6E4409CDEEF347279394F32430E60EDC66095A9872F4DB |
| step000016.json | E532058AF257B94AC176B9012BA5D031FE926681BC809C038BBB2B02543CF386 |
| step000032.json | E79F979C1A6EA599654F77F94EC70B19FC5A529C348A7B7939ED2998B78FF70E |
| step000048.json | 4BD509C151FE4A09A84CDC02FDAB46F56D98E07D673EDF4960C3809661864D42 |
| step000064.json | 85CA3C83F2075DD8333A8D4B680C9CC6897234CDE87A10A68F6C02D695FAA291 |
| step000080.json | 61D0E7A71FA01DFC2B907D42EFBEA7A494C11EB8927D321A370BFF7B8F512AA0 |
| step000096.json | E2562DF4CD5F347BE99D1B16D1B6C76205D90953A5C4EEFC967DC04C9FFF9D39 |
| step000112.json | 44699F5AB5A38CBAB2D8F45BC8199213829CB143648EF3C0CB3F56717BEEEEC8 |
| step000128.json | 0441B4715D72F8E5CBC0C83C993323B0038DAD42920B561A4E7AB249E921EE5E |

### 7.2 数值、身份与结论范围

结论稿表格的稳定 5/8 s risk、失稳真实区间 risk 与两个时域漏判计数，逐行与公开对象对应一致（六位小数舍入）。全部 11 对象 status=COMPLETED、all_required_rows_qualified=true，两个分类时域数值未决均为 0、误报均为 0，分母为 267 稳定/30 失稳。八个 fresh_rollouts=true，三个复用对象为 R_N、A96、step128；各点 optimizer=0、val/test=false、新 FOM=0。

step16 是九点稳定 5/8 s 风险最小点，但真实区间 risk=4.0964692967、漏判22/30，分别未过 R_N risk 及真实区间漏判保护。step96 真实区间漏判20/30，两个参照各新增漏判0、修复漏判1，支持“保留原9个检出、修复1个”；其8 s相对A96新增漏判1、修复0，漏判4/30，因此仍未过原门及新身份守卫。九点不通过已经由原 efficiency_target 决定，不是只因本轮前瞻新增身份/T 守卫而失败。

保护集合计数8 s=27、真实区间=9，固定T=9且truth一致。step16/48/64/80的T覆盖门失败、E/u均null；其余五点E/u及两个参照E/u与结论稿相符。没有用缩小后的交集宣布时机改善，也没有把固定9事件的时机结果扩大为30事件结论。eligible_checkpoints与pareto_front均空、suggested_adaptation_source=null，科学状态unresolved。

结论稿将来源审计与当前 optimizer 不适用分开，保留五点原64步健康检查通过、四点insufficient_history及充分性unknown；符合§6审查边界。已暴露train、单一优化日程与非独立确认均明确，未扩展为A结构族否证、泛化结论或机械p结构采用依据。

### 7.3 唯一文案修正与三个决定

初读结论稿“1/2/5/8 s风险……完整保存在公开逐点JSON”与实际公开字段不一致。公开摘要仅含稳定5/8 s风险、8 s共同转速RMSE、8 s归一化p RMSE及分类状态；应按这些确有字段表述。若另指1/2 s原逐事件轨迹指标，应明确另一个实际存储位置，并避免把轨迹指标称为本公开摘要已有的risk。本 Reviewer 已将此具体修正反馈root，不因此要求新增计算。

代码retain_branch、模型retain_evidence及科学unresolved可以同时成立；执行完成不等于能力目标通过。H3的支持范围限于“既有路径上稳定精度受检查点位置影响”，不能直接变为优化充分性或新增结构必要性。引用的下一p端口材料标DRAFT_UNADOPTED，正文明确未登记新Run、未计算、未采用结构；本审查不为该草案背书，也不创建新Run。训练未恢复、无晋升、所有模型/工作指针不变是本轮决定的范围。

### 7.4 执行回执与科学状态分别判断

公开execution_receipt记录stage1/stage2/decision退出码均0、decision只执行一次，17个登记PID身份检查均为pid_exists=false、this_run_active=false，active_owned_process_count=0。此处确认的是回执字段一致性；本Reviewer未另行查询当前系统进程，因此不把公开回执说成自己实时复核了操作系统。

Stage2 controller wall=562.8700891 s，来源审计wall=2.886849 s；stage1两参照新增审计耗时18.8386081+15.7028993=34.5415074 s。决策process wall=null，工具等待40.2020624 s单列，未拿等待时间填充进程wall。并发单点耗时、controller经过时间和旧训练/推理成本未混加；CPU总进程秒仍unknown，未作同精度加速主张。公开回执没有逐尝试终态列表，本Reviewer不把结论稿“一次完成、无retry”另称已独立核验过私有每次尝试。

因此执行层为已登记阶段完成且已记录无本Run活动进程，科学层仍为本九点无全门点、目标未解决。两者分别陈述，未用退出码0、COMPLETED或本轮工作量完成替代科学判断。

## 8. 最终文案与公开回执整理复核（2026-10-30）

**最终审查决定：通过，§7 的公开指标位置说明已修正关闭，无剩余结论阻断。** 本次追加前本报告 SHA256 为 `27EFF970871572F9FE2347494C4E7C28FFB1735CCE020C87608EC4817C13257B`。

结论稿现 SHA256 为 `7B97DC1915B000791C6F4CB09B2F4F4CF9567524B7275FD397F170FC65DB9182`。已实际复读修正段：公开文件只称保存稳定5/8 s风险、8 s共同转速RMSE、8 s归一化p RMSE及分类计数；1/2 s逐事件轨迹指标另列为私有原评价evidence/rows，未再声称其存在于公开风险摘要。该修正与所审评价源码的指标/产物分工一致，未改变任何计算或门槛。

同名公开 execution_receipt.json 现 SHA256 为 `2C74ECB470465C306DDFF12BFB58C927763BC2B6BC162A171019DEC88D7DC662`，已实际读取：删除原生PID/session/创建身份明细，仅保留阶段退出码、wall/工具等待、17个身份已检查且全部退出、活动进程0、optimizer/val/test/FOM状态与私有原件哈希。三个阶段退出码0、stage2 wall562.8700891、decision process wall=null、工具等待40.2020624等值与整理前一致。

公开 `private_original_sha256` 为 `072C9D2FF968CB0B87C181683D39E8F63B47E94773BF32B011F55D3A734F7E92`，恰与本Reviewer在§7实际读取整理前公开原件所得哈希一致。root记录已将原字节保存为私有 execution_receipt_original.json；本Reviewer未打开该私有副本，因此确认的是原件摘要连续性与公开结果不变，没有虚称独立复核私有复制操作。历史§7保留旧回执哈希，本节记录公开整理后的新版本，避免把整理当作科学结果变更。

可归档的最终判断保持：代码retain_branch，模型retain_evidence，无建议来源/无晋升，科学unresolved；执行已记录完成和退出，训练未恢复，原val/test关闭、新FOM=0、TASK4暂停。下一p端口材料只作为DRAFT_UNADOPTED引用，不据此采用机制或登记新Run。独立审查限于冻结源码、人工样例和公开结果投影，不等同私有预测重算。
