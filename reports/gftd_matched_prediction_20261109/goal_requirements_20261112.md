# GFTD 三步主线完成要求与已核证范围（2026-11-12）

有限 K4 静态恢复、调制模型的因果接口，以及原指数调制必须收缩全域存在主张的数学证据已经成立。有界核在已审假设下解决了有限时域存在性问题，并通过原 train 297 包的同核初始化检查。**整项目标仍待闭合**。调制恢复在原人工条件下未达到目标，当前有界模型的联合辨识为 unknown。两模型的同期 fit 已冻结，完整 594 个组合的 L1、L2 和 L3 待最终证据。目标和口径以[三步主线](../../designs/gftd_mainline_20261107_architecture.md)、[项目合同](../../kiki.md)和[当前协议](../../campaigns/gftd_matched_prediction_20261109/protocol.md)为准。

| 要求 | 状态 | 已核证的范围与结果 |
|---|---|---|
| Run 1：从候选完全图 K_N 构造边力字典，处理未观测 p、r | 明确范围内已证实 | 一般 K_N 的候选字典定义与有限恢复验证分别判断。在 K4 六边上，从转速历史和真实末点 p 重建完整机械功率，重建力记忆受迫项，并以四个弱测试函数积分构造 A_edge。用 SVD 投影 I−P_Z 消去逐事件记忆初态，再按绝对激励、逐列条件信息和秩判据筛选，有限枚举与非负最小二乘给出去偏权重。真 p、r 历史与真图只用于冻结后核对。平衡条件的 24 个设置均为 null，说明这些条件下信息不足。[Run 1 协议](../history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_protocol.md) · [结论](../history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) |
| Run 1：找回未知支撑和边权，假边权归零 | 明确范围内已证实 | 登记的零图、路径、星图和环图无噪支撑全对，假边和漏边均为 0，最大边权绝对误差 2.0612e-4。固定噪声方向的两档支撑仍全对，较高噪声档最大边权误差约 0.264。完整枚举结论限于 K4 六边的64种候选支撑及登记激励范围。有效图不等于线路图，Kron 对照不作为有效支撑或权重真值。当前 K10 完整库的条件信息支持列可区分性，IEEE39 有效支撑与权重的联合恢复证据不足。[Run 1 结论](../history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) · [当前 fit 审查](fit_review.md) |
| Run 2：由相位势 ψ 经储能梯度驱动隐幅值，调制反馈使用同一 5N ODE | 明确范围内已证实 | 0≤S_mod≤S_edge≤1 的层级约束通过非负系数及门映射实现。节点 gamma 与幅值尺度规范按原主线冻结，属于先验而非已恢复参数。相位力使用调制后的边系数，幅值方程保留 β K′(h) ψ 的反作用，静态嵌入保留。完整 fit、save/load、initialize 与自主预测已在人工 K4 闭环。simple600→smooth1200 按合法因果窗估计隐初态；48/48 数值合格，30/48 达到登记 1 s 预测门。有界核另有同核存取、梯度和初始化资格。[Run 2 结论](../gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md) · [有界核结论](../gftd_bounded_initialization_20261109/conclusion.md) · [当前 fit 审查](fit_review.md) |
| Run 2：恢复未知静态与调制支撑、权重及隐幅值，并辨别静态和记忆替代解释 | 与原人工恢复主张相反；当前机制证据不足 | 原无噪调制族候选退到 β=0，最大 β 误差 0.55、调制支撑 null。含噪调制族漏掉 3 条真调制边。无噪调制族两方案都通过 1 s 预测门，预测通过无法区分调制与替代解释。当前 24 包主细网格的静态 45 列与有界 90 列各自通过条件信息门；调制侧固定了已推断幅值历史。两模型各保留父参数，各三次提案对应同一个候选，最终均为 not_identified，support=null，联合辨识和优化充分性仍为 unknown。[Run 2 结论](../gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md) · [当前 fit 审查](fit_review.md) |
| Run 2／3：原无界指数在耗散下可对所有有限初态延拓 | 与该全称主张相反；有界修订的存在性已证实 | 两节点精确反例中 H 下降且幅值有限时间逃逸。固定 DP8 将原 RK4 的 47/594 初始化资格提高到 579/594，独立参考在五个保存状态的共同窗口仍证实幅值增长。K(h)=exp(tanh(h)) 保留零点二阶展开及静态极限，在有限参数、正质量、正局部储能和正时间常数，且耗散门为原常数或有界并局部 Lipschitz 的条件下，完整5N向量场局部 Lipschitz、至多线性增长，因此精确解在任意有限时域存在。该结论不自动保证长期有界、局部吸引域覆盖、浮点积分资格或预测准确。两组旧初态通过 1 s 三参考一致性，有界同核初始化 297/297 六门通过，严格 1 s 确认 289/297。存在性、长时数值资格与真实预测分别取证。[精度结论](../gftd_initialization_precision_20261109/conclusion.md) · [长时诊断](../gftd_long_horizon_diagnostic_20261109/conclusion.md) · [储能证明](../gftd_storage_feasibility_20261109/conclusion.md) · [同核初始化](../gftd_bounded_initialization_20261109/conclusion.md) |
| Run 3／L1：原 train 297 上同期比较完整自治预测 | 未完成 | 原 RK4 轮静态 46/297、调制 1/297 合格，失稳各 0/30，原分母总体 L1 为 null。当前两模型共享合法输入与匹配适配规则，实际成本分别记录。冻结首批 103 个组合全部通过六门并有完整 15 s 主细轨迹，102 个完整严格参考确认，1 个到限。首批覆盖 52 个静态、51 个有界调制组合。完整同期效应仍为 unknown。历史 R_N 漏检 21/30 沿用历史模型与口径。[原 Run 3 结论](../gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md) · [历史结果](../history/force_memory/originals/memory_transport_20261029_conclusion.md) · [首批审查](deploy_batch01_review.md) |
| Run 3／L2：对最终同一部署 ODE 给出局部吸引域和初态裕度 | 未完成 | 原 Run 3 的冻结静态及旧指数调制各有非空局部吸引集，实际返回初态获证分别为 0/46、0/1，均在严格球外，点裕度未算。当前有界核的全域存在证明与该局部证书各自成立于相应假设；当前最终 ODE 的证书及初态覆盖待结算。[原 Run 3 结论](../gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md) · [当前协议 §7](../../campaigns/gftd_matched_prediction_20261109/protocol.md#7-最终ode的l2严格局部证书) |
| Run 3／L3：最终 ODE 对照 Kron 与 13 个物理参考模态 | 未完成 | 原 Run 3 的有效耦合与冻结 Kron 导纳已有图对应结果，两模型各配对 10/13，保留未匹配项；当前有界模型须按同一冻结参考给出本轮结果。L3 按原协议报告，无最低配对门。[原 Run 3 结论](../gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md) · [当前协议 §8](../../campaigns/gftd_matched_prediction_20261109/protocol.md#8-l3与实际平衡) |

## 闭合所需的现有证据

- **同期预测。** 冻结两模型各 297 包的完整端点选择，核对因果输入为 11 点、100 ms 的相角和转速历史、真实末点 p₀ 及冻结物理常数，后续轨迹由各自共享自治 ODE 产生。按原协议一次性评分，报告稳定 267 的 5 s risk 和 1/2/8 s 结果、共同转速 RMSE（rad/s）、归一化 p RMSE，以及失稳 30 的 8 s 和真实可用区间分类与漏检、误报、未覆盖、未决和共同可比触发时机。risk 先对每个事件的归一化点平方误差作时间平均，再取事件均值的0.8倍加最差向上取整10%事件均值的0.2倍。保留原分母、原数值资格和失败成本；任一 eligible 失败按合同记相应 official=null。报告可据此判断固定两套方法在本次 train 的实际差异。[当前协议 §6](../../campaigns/gftd_matched_prediction_20261109/protocol.md#6-完整初始化参考和真实l1)
- **稳定与物理对应。** 对同一最终 ODE 结算 L2 的平衡、局部曲率、紧致正定区域、不变性与严格下降或 LaSalle 条件；逐点报告所在连通分支及 m=1−V(z₀)/ρ 或未获证，明确报告初始化误差敏感性；确定性误差集或相应结果缺失记 null，不以优化残差代替误差球。L3 对当前两个最终模型分别按 report-only 结算，报告实际平衡、声明 Kron 口径、13 个冻结模态的一对一配对及未匹配项，保留频率、阻尼和振型相似度。证书适用对象是最终 ROM。[当前协议 §§7–8](../../campaigns/gftd_matched_prediction_20261109/protocol.md#7-最终ode的l2严格局部证书)
- **恢复主张与方案调整。** 支持恢复需要证据能在合法观测下区分共享边参数、幅值和力记忆的补偿，并按已登记的真／假边和参数口径核对；可确定范围或等价类也须给出明确条件。原指数的反例已经证实该全称存在性主张需要调整。有界核修订后的机制恢复与预测效应由各自证据判断。若完整同期结果和独立审查把未达目标定位到明确的数值、初始化、搜索或可辨识条件，可据该条件提出有范围的方案修订。原人工恢复失败、重复提案未获接受和条件字典满秩各支持表内范围的判断。当前无据把整项目标关闭为完成或整体路线否证。[三步主线](../../designs/gftd_mainline_20261107_architecture.md) · [当前 fit 科学判读](fit_assessment.md)

## 成本与权限口径

状态维数、可训练参数量和固定结构先验分别报告；保留初始化、积分、总耗时，以及调优和失败成本。同精度成本比较须使用相同数据、设备、精度、步长、时域和批量的实测结果，缺失时写“未比较”，不能从参数量推算加速。

原始 val/test 关闭，旧已消费 test 永久退出选择，新 FOM 为0，TASK4暂停，default none。train385与noise88沿用既有报告权限，不由本清单扩展数据读取或选择权限。当前真实train未来依原协议，在两模型与全部594个初始化选择冻结后才由独立评分阶段读取。

## 关键证据的字节身份

| 来源 | SHA-256 |
|---|---|
| [三步主线](../../designs/gftd_mainline_20261107_architecture.md) | `0459ef7f11293a24d8190da0157bef621aa99399ed346966f4536a0800c08a2c` |
| [项目合同](../../kiki.md) | `b2b2490b7b3d3ee9ea318c4bade39f4e918b59b96f844acd7555541a72e0ac19` |
| [储能独立设计审查](../../reports/gftd_storage_feasibility_20261109/design_review_v2.md) | `84d783f046ce83547291ebb02cd7168513c6b5631e6e0341454d0e4a619544b4` |
| [Run 1 结论](../history/gftd_campaign/originals/gftd_sparse_topology_20261107/gftd_sparse_topology_20261107_conclusion.md) | `4d74cc22d384e23536d508ddd5596fa7d16f6de094f65f60ff33d24d83b4c081` |
| [Run 2 结论](../gftd_modulation_20261107/originals/gftd_modulation_20261107_conclusion.md) | `2e4d6c529fd215991a150815cd962e7093d7b5920e8299b0495297e6910a736c` |
| [原 Run 3 结论](../gftd_fit297_20261107/originals/gftd_fit297_20261107_conclusion.md) | `1508bd817ea510b207159fbf6f070c06cb24f65c2f46a085d0c87d3a4821b78c` |
| [精度结论](../gftd_initialization_precision_20261109/conclusion.md) | `ba142fa5f1bc928265c3a9ec863717e9a9ceb5f1f58ea34c86216d163482a766` |
| [长时诊断](../gftd_long_horizon_diagnostic_20261109/conclusion.md) | `15ff8218b30d18224c1672664d096d7e69367689141dc2fadaad67225699949f` |
| [储能证明](../gftd_storage_feasibility_20261109/conclusion.md) | `609926b7ba00d5974b49c424e5516a2a890ed0d527084fa1ebcb0cb65bf8d134` |
| [有界初始化结论](../gftd_bounded_initialization_20261109/conclusion.md) | `d233e052aeacb31255502634b823c2cb0d677ad205f922c488b21b0782322781` |
| [当前 fit 审查](fit_review.md) | `8f59580fd8badbd950f29dfcac9e20717aefb4c12d62c9adc9649807dd1d407d` |
| [首批 103 组合冻结清单](../../results/gftd_matched_prediction_20261109/deploy_batch01_audit.json) | `8ddcf0eabcc56c203cbce775c2067e71cc20551a5530b43527a7e2f6e94e2961` |
| [当前协议](../../campaigns/gftd_matched_prediction_20261109/protocol.md) | `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43` |

本清单按[独立审查](goal_requirements_review.md)补齐范围、假设和评价口径，只整理已保存的报告、协议和证据身份。原始输入和未来读取为 0，新增模型、ODE、H、SVD、优化和测试调用为 0。最终科学采用待完整阶段结果与独立审查。
