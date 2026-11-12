# 当前研究状态

**2026-11-12：匹配预测 Run 已完成 fit、594次部署、L2/L3和唯一一次真实train评分，并通过独立审查。两模型各297/297通过初始化六门；调制在稳定5/8秒有3个数值未决，原分母主项为null。保留实现与证据，不晋升模型；整项目标仍未完成。**

## 当前范围与证据

- 活动Campaign：[GFTD结构发现](../campaigns/gftd_kuramoto_discovery_20261106.md)；[架构入口](current_design.md)。
- 本轮：[协议](../campaigns/gftd_matched_prediction_20261109/protocol.md)、[登记](../harness/runs/gftd_matched_prediction_20261109.json)。执行COMPLETED，工作量closed；所有宿主及worker已退出。
- [科学判读](../reports/gftd_matched_prediction_20261109/final_assessment.md)、[独立L1审查](../reports/gftd_matched_prediction_20261109/l1_review.md)、[主张审查](../reports/gftd_matched_prediction_20261109/final_claim_review.md)、[root最终决定](../results/gftd_matched_prediction_20261109/final_adoption.json)。
- [全量部署摘要](../results/gftd_matched_prediction_20261109/deploy_digest.json)、[L1摘要](../results/gftd_matched_prediction_20261109/l1_digest.json)绑定原分母、逐事件身份、主细资格及实际成本。
- 代码retain_branch，模型retain_evidence，科学continue；两模型均保留父参数与not_identified，联合辨识和优化充分性unknown。
- 科学源码提交a1936bc5，43份源在闭合时仍与prepare_1一致。[原字节档案](../results/gftd_matched_prediction_20261109/source_archive.json)保持不可改；本轮源冻结已在明确阶段边界解除，新执行仍须另行冻结和审查。

## 本轮主要结果

| train范围／指标 | 静态 | 有界调制 |
|---|---:|---:|
| 初始化原六门 | 297/297 | 297/297 |
| 完整1秒严格参考确认 | 297/297 | 289/297，另8个到限未决 |
| 稳定267，5秒risk | 12.312022 | null，264/267合格 |
| 稳定267，5秒共同转速RMSE（rad/s） | 0.545177 | null |
| 失稳30，真实区间risk | 9.612748 | 11.805068 |
| 失稳30，真实区间漏检 | 21/30 | 17/30 |
| 失稳30，8秒分类漏检 | 7/30 | 7/30，身份不同 |
| L2正式初态获证 | 0/297 | 0/297 |
| L3参考模态配对 | 10/13 | 10/13 |

risk是逐事件时间平均归一化平方误差的加权风险，不是RMSE。30个失稳样本从100ms部署点后的真实评分时长为0.03–1.44秒；15秒是协议上限。调制新增检出4个失稳样本且保留静态原9个。静态3个误报恰为调制3个数值未决，不能宣称误报修复。共同可比9例中，调制6例更早但绝对时机误差更大，其余3例相同。

调制5/8秒未决为expansion_0062、expansion_0094、pilot_0255；前两例自身5秒主细通过，因原联合8秒门失格，第三例自身5秒也不合格。全部保留原分母。两模型局部证书均成立，但全部实际初态在证书区域外，初始化误差界仍为null。

fit与部署科学RHS共280,219,557，人工资格另28次RHS、748次H；评分新增模型调用0。部署宿主耗时4,753.69秒，评分77.41秒；FOM同精度成本未比较。原始模型、观测及轨迹留本地。

## 下一授权动作

按[三点草案](../designs/gftd_objective_alignment_20261112_draft.md)与[设计审查](../reports/gftd_matched_prediction_20261109/next_design_review.md)，补齐实际系数／支撑、逐次评价身份和固定父端点诊断口径，再审查是否登记一个有限历史目标对齐Run。只检验现有弱式方向在1/2、1/4、1/8三个更新幅度是否有合格改善；全部失败即停止该方向，不加更小步长或重复轮数。当前尚未登记或释放新Run，不自动接另一轮全量L1。

[原三步目标要求](../reports/gftd_matched_prediction_20261109/goal_requirements_20261112.md)仍有效：有限K4静态恢复已验证；一般拓扑与隐调制联合恢复、预测能力目标、实际初态L2覆盖仍待完成。原val/test关闭，旧test退出选择，新FOM=0，TASK4暂停，default none及protected anchors不变；部署仅100ms因果历史和末点真实p₀。
