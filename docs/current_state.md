# 当前研究状态

**Layer 1 可辨识对照。** 同步平衡与纯幅值初态对 56 个边参数的灵敏度均为零。已知总转矩的同频补偿构造中，12 条边与 11 条边产生相同观测。角激励和波形重合度相图定量区分满秩、近简并与精确等价。开放功率账、线性增长证明和有限时间逃逸复现已归档。Layer 1 正式拟合仍按资源预算排程。[可辨识证据](../reports/osgkd_v2_wp2_layer1_20261127/identifiability.md) · [方法与理论](../paper/osgkd_v2_methods_and_theory.md)

**Layer 1 长窗资源预算。** 18 格合成前后向测量通过，三个规模的 4 s 物理批量均可取 4。最大规模 $N=32$、496 条候选边的 4 s 前后向中位耗时为 3.588 s，峰值工作集为 1.645 GiB。参数更新、新 FOM 与 IEEE39 `test` 事件轨迹访问均为 0。WP1A 进入 24 组串行重拟合，Layer 1 参数拟合等待 WP1A 结案。[预算表](../reports/osgkd_v2_wp2_layer1_20261127/resource_budget.md) · [Layer 1 执行协议](../campaigns/osgkd_v2/wp2_layer1_execution_protocol.md)

**WP0 零更新诊断（2026-11-27）。** 三对边贡献秩相关为 0.904、0.818、0.912，均通过节点置换与 Holm 校正。Top-30、Top-20 满足一致性门，Top-12 一对 F1 为 0.667。对 `pilot_0232` 的边速差尺度干预使两个误报种子的 1 s 误差升高 35.6%、61.4%，WP1B 未触发。WP1A 用 Top-30/20 和四项模块对照，先核验固定同种子配对。九个变体的叶参数优化器回归已通过。[WP0 报告](../reports/osgkd_v2_wp0_20261127/result.md) · [独立复核](../reports/osgkd_v2_wp0_20261127/review.md)

**Sprint 2B-1 结案（2026-11-26）。** 两组各接受 2048 次更新、零拒绝，固定末点均通过原始 `val` 双步长核验。Neural ODE 稳定 8 s 风险 0.1584、自主召回 0/12，封闭被动 Kuramoto 为 2.3462、5/12，两组误报均 0/95。同种子开放扁平 OSGKD 为 0.2200、10/12、0/95。两个 Sprint 2A 图种子的唯一误报均为 `pilot_0232`。Layer 1 的 N=32、496 条候选边、批大小 8 短窗前后向峰值为 787.6 MiB。备战包 A 和 Layer 1 正式训练暂停。[Table 2](../reports/osgkd_v2_sprint2b_baselines_20261126/result.md) · [独立复核](../reports/osgkd_v2_sprint2b_baselines_20261126/review.md) · [运行登记](../harness/runs/osgkd_v2_sprint2b_baselines_20261126.json)

**Sprint 2A 结案（2026-11-26）。** 四组各接受 2048 次更新，零拒绝。原中断的三段训练记录逐项重放一致。四个固定末点均通过原始 `val` 双步长核验。图组三种子稳定 8 s 风险为 0.8979 ± 0.4051，8 s 自主召回均值为 8.67/12，误报为 1/95、0/95、1/95。图主干未达到晋升条件。同种子扁平 `24111` 的风险为 0.2200、误报 0/95、自主召回 10/12（原始分母 10/22）。四个末点作为证据保留。Sprint 2B 和 Layer 1 暂停，等待确认启动顺序。[Table 1](../reports/osgkd_v2_sprint2a_20261125/result.md) · [独立复核](../reports/osgkd_v2_sprint2a_20261125/review.md) · [运行登记](../harness/runs/osgkd_v2_sprint2a_20261125.json)

**Sprint 2B 和 Layer 1 备战（2026-11-25）。** 三种子剪枝和删组重拟合、等变图 Neural ODE 与封闭被动 Kuramoto、三个规模的合成真值及两机逃逸复现均通过合成资格和独立实现复核。正式科学工作按前瞻方案登记。[资格报告](../reports/osgkd_v2_discovery_preparation_20261125/result.md) · [Sprint 2B](../campaigns/osgkd_v2/sprint2b_discovery_baselines_protocol.md) · [Layer 1](../campaigns/osgkd_v2/sprint3_layer1_protocol.md)

**图模型末点探针（2026-11-25）。** 第 1792/2048 次去共同角后的 49 维谱最大实部为 +0.3668/+0.3628 s⁻¹，各有一个正实增长模态。第 2048 次触发谱铰链分支。固定误报在 2–4 s 已扩大，起报 5.51 s 触发。完整 `train` 的同状态作用中，环境记忆读出、基准边和调制合计为 0.568、0.0597、0.0264。[探针报告](../reports/osgkd_v2_p2a_post_audit_20261124/result.md)

**等变图初始化器（2026-11-25）。** 从完整 `train` 从头联合训练 2,048 次，全部接受。节点置换、相角平移、零参考与同权跨规模门禁通过，编码器为 1,928 参数，比扁平模型少 68.77%。两个固定点的 `val` 自主失稳检出均为 9/12（9/22）、误报均为 1/95，稳定 8 s 风险分别是 0.881 和 0.939，未达 0.55。误报为同一条稳定样本，主报告点第 2048 次与第 1792 次均保留为证据。[本轮报告](../reports/osgkd_v2_p2a_equivariant_graph_20261124/result.md) · [运行登记](../harness/runs/osgkd_v2_p2a_equivariant_graph_20261124.json)

**封顶基线和活动口径（2026-11-24）。** 原方程第 2260 次是 OSGKD v1 扁平 GRU 50 维封顶锚点。`val` 稳定 8 s 风险 0.427074、误报 0/95，自主失稳检出 7/12（原分母 7/22）。失稳真实区间计数和触发延迟作为诊断。旧框架的 60 维零和快记忆与高阶势作为探针封存。[评价修订](../campaigns/osgkd_v2/evaluation_amendment_20261124.md) · [锚点报告](../reports/osgkd_v2_sprint0_20261124/result.md)

**后续数据口径（2026-11-23）。** IEEE39 使用完整 `train` 的 385 条（342 稳定、43 失稳）训练，原始 `val` 的 117 条（95 稳定、22 失稳）验证。其中 10 条 `val` 失稳样本不足 100 ms，计入原分母并标明未覆盖。新训练比较使用完整 `train` 建立批次顺序，用相同的双步长 `val` 验证规则比较各组。沿用冻结的父模型尺度。`test` 关闭，新 FOM 为 0，TASK4 暂停，default none。[数据合同](../kiki.md#3-数据与信息) · [数据入口](../src/kiki/data/osgkd_train_val.py) · [验证入口](../src/kiki/evaluation/osgkd_validation.py)

**冻结基线和完整 train 对照（2026-11-24）。** 第 1860 次父点在稳定 8 s 的 `train`、`val` 风险为 0.756、1.757。固定方程各完成 200 次接受更新后，`val` 稳定风险为原目标 0.842、分段风险及分层采样 1.045，两组均误报 0/95。候选降低失稳真实区间运动风险，两组真实区间检出仍为 `train` 3/43 和 `val` 0/12（原分母 0/22）。该轮按原协议保留两端点作为证据。[冻结基线](../reports/osgkd_v2_train385_baseline_20261123/result.md) · [配对结果](../reports/osgkd_v2_train385_risk_alignment_20261123/result.md)

**2026-11-23。** 当前方案为 [OSGKD 2.0](../designs/osgkd_v2_architecture.md)。[P0 零训练诊断](../reports/osgkd_v2_p0_20261123/result.md)、[P1a 固定末点评价](../reports/osgkd_v2_p1a_20261123/result.md)及[P2a 初态试验](../reports/osgkd_v2_p2a_20261123/result.md)已完成并独立复核。P1a 两边各 200 次成功更新，主细步长评价覆盖全 297 条。

- P1a 的 2 s 与 8 s 训练组，在稳定 267 条的 8 s 运动风险为 39.751 和 2.038，稳定误报为 15/267 和 0/267。8 s 组 1/2 s 风险较高，保住原 C1 三个真实区间检出，仍有 27/30 漏检。[P1a 登记](../harness/runs/osgkd_v2_p1a_20261123.json)
- P2a 固定 P1a 长组方程，一步精修使历史平方残差降低 98.3%。稳定 8 s 风险由 2.038 升至 89.481，稳定误报由 0/267 升至 105/267，失稳真实区间由 3/30 增至 4/30。精修未通过采用守卫，保留原初始化器和长组方程。[P2a 登记](../harness/runs/osgkd_v2_p2a_20261123.json)

- v1 C0/C1 固定参考均有局部增长。去共同角、保留共同转速后的闭环谱最大实部分别为 1.44007 和 1.47467 s⁻¹。局部增长与实际长期预测误差需分开检验。
- C1 在失稳样本真实记录内检出 3/30，26 次触发发生于真实记录结束后，另有 1 次截至 8 s 无触发。27 条真实区间漏检中有 13 条的幅值初态达到边界的 99%。[P0 回执](../harness/runs/osgkd_v2_p0_20261123.json)

**前序证据（2026-11-20）。OSGKD 首轮完整 C0/C1 训练、一次条件适配及全 train fit297 正式评价已完成，最终报告已通过独立复核，证据与工作量已闭合。**

部署为 [motion_only_v1](../kiki.md)：11点/100 ms相角与完整转速历史，真实p0不输入。该轮采用 [OSGKD v1](../designs/osgkd_architecture.md)，协议为 [fit297首轮](../campaigns/osgkd_ieee39/first_fit297_protocol.md)。

- 两组各1500次更新、0非有限拒绝，完整事件顺序和时域配对。均按原规则追加500次，并在最终评分前选定1500次检查点。C1未静态回退；参数6809/6854，状态均50。
- 两模型全部297的8 s主细数值检查通过，最大归一化RMS差分别2.70e-8、9.39e-8。当前大预测误差不能用本步长范围内的积分精度不足解释。
- train稳定267的8 s risk：C0 78.28062、C1 42.60901（低45.57%）；共同转速RMSE：1.47912、1.05578 rad/s。稳定误报60/267→6/267，C1的6例是原60例的子集。
- 稳定1/2 s的C1 risk高18.11%/14.33%；失稳真实区间两模型均只检出3/30、漏检27/30，内部检查12例均漏。8 s的27例未知尾段保留未覆盖，不算成功检出。
- 原来源审计保留blocked/unresolved；5个历史子窗连续闭合未决。端口按经审查的离散观测合同近似监督，样本全部保留。

该轮三个决定为代码 retain_branch、模型 retain_evidence、科学 continue。工作候选为 C1 第 1500 次，C0 同合同对照同步保留。该轮原始 `val/test` 关闭，新 FOM 为 0。

[早期闭合诊断](../reports/osgkd_v2_early_closure_20261123/result.md)选出总电气端。[固定末点配对适配](../reports/osgkd_v2_early_electrical_20261123/result.md)各 32 次更新后，候选组原 27 条漏检的短窗电气与运动均方误差相对对照组分别下降 1.71% 和 3.43%，稳定 8 s 风险为父模型 2.038、对照组 1.115、候选组 2.075。三组均保有原 3/30 检出和 0/267 稳定误报。

[固定端点诊断](../reports/osgkd_v2_conflict_profile_20261123/result.md)在稳定变差组所抽 4 条上发现两端点重复的耦合、环境局部梯度冲突。原 27 条漏检有 22 条的主要加速机组持续高估电气作用，三个端点幅值初态触界均为 0/27。优先检验单一高阶耦合势。

[学习率对照](../reports/osgkd_v2_p1a_clean_parent_20261123/result.md)各完成 128 次更新。train 稳定 8 s 风险为父点 1.115、原学习率 0.429、0.3 倍 1.007。两组稳定误报 0/267，保有原 3/30 检出。采用原学习率第 1860 次作为研究父点。

[高阶耦合势对照](../reports/osgkd_v2_p1b_electrical_unlock_20261123/result.md)已通过真实零点资格并完成 160 对。原 27 条漏检的 0.25 s 真实截断窗，高阶势组的总电气和运动均方误差比原势组低 4.97% 和 7.27%，检出为 4/30 和 3/30。稳定 8 s 风险为高阶势组 0.750、原势组 0.605，双方均 0/267 误报。联合采用门未通过，保留第 1860 次父点。

[电气误差分解](../reports/osgkd_v2_electrical_decomposition_20261123/result.md)已完成复核。第 1860 次父点在原 27 条漏检的两个短窗中，物理零和校正残余下界为 5.84% 和 3.17%。选定机组相邻边均远离调制底板，三个分流条件均未通过，保留工作父点。[v2 Campaign](../campaigns/osgkd_v2/campaign.md)

[本轮报告](../reports/osgkd_fit297_20261113/result.md) · [机器结果](../results/osgkd_fit297_20261113/analysis.json) · [科学综合](../reports/osgkd_fit297_20261113/scientific_synthesis.md) · [Run记录](../harness/runs/osgkd_fit297_20261113.json)
