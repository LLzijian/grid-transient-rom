# 当前研究状态

**2026-11-07：用户已明确指示“按照此方案开始研究和推进”，GFTD 修正与研究推进恢复。**
已修正联合储能与独立机械功率状态，完成首轮人工局部观测探针；范围内推进无需逐轮再次请示用户。

## 当前范围与指针

- 原始 val/test 关闭；旧消费 test 永久退出选择；新 FOM=0；TASK4 继续暂停。
- 部署仍为 11 点、100 ms 因果 δ/ω 历史与末点真实 p₀，随后共享自治 ODE。
- default none；incumbent 与 protected anchors 保留原模型及原适用范围。新 GFTD 是工作候选，未晋升。
- 现有 train fit297 暴露谱系不变；人工算例仅支撑其明确生成条件下的结论。
- 授权及父归档见 [authorization](../results/gftd_repair_20261107/authorization.json) 与 [parent_archive_reconciliation](../results/gftd_repair_20261107/parent_archive_reconciliation.json)。

## 已核实事实

前轮 p_constant_norm_20261031 已完成普通归档。2026-11-07 再核远端 codex/force-memory-20261028 为 bd4db257daf6f1bc925880a996f717d401faae6b，与本地历史回执一致；33 个公开归档文件原字节匹配。无须重放该 Run。

该轮仍只证明固定父参数、已发布 binary64 范数与精确整数门限下，六机共享常数必要条件交集为空。它不排除重新拟合模型族、其他合法初始化或优化改善，也不识别真实物理失配原因。常数支线及旧工作量余额继续关闭。

GFTD 原有 6 个单测通过，但未证明旧 40 维基线嵌入、全状态域功率恒等式、COI 动力学解耦或部分观测结构恢复。系统评估已给出指数截断梯度和 COI 交叉作用反例。当前任务优先消除这些实现与命题不一致。

## 当前执行与下一动作

gftd_repair_20261107 已完成独立设计/实现审查及 43 项冻结资格测试，采用 5N 独立 p/a/r 和精确父映射。另有执行后的[两图梯度/功率补充检查](../results/gftd_repair_20261107/power_supplement_receipt.json)：16 个状态 × 2 门共 32 项通过，单独保留原释放时的覆盖范围。非零调制部署初始化与部分观测拟合尚未实现，不晋升模型。

gftd_observability_20261107 已完成，唯一 action 为 gftd-observability-20261107-01，8 个固定三节点条件 × 6 种数值设置全部完成，耗时 76.36 s。三个开启调制的非平衡条件均为局部 7 秩，条件数约 124 万—182 万；平衡与关闭调制的负校准成立。此结果限于登记尺度下的局部信息，不证明噪声下恢复或真实系统收益。详见[科学判读](../reports/gftd_observability_20261107_scientific_assessment.md)和[独立结果复核](../reports/gftd_observability_20261107_result_review.md)。

代码决定 retain_branch；模型决定 retain_evidence；科学决定 continue。本次有限工作量关闭，不转移余额。下一项工作是先冻结观测误差和未来误差尺度，再设计已知共享参数下的合法隐状态初始化比较；固定共享调制参数，比较 a=0 简单初始化与 a/r 推断，检验噪声敏感性。新 Run 尚未释放，IEEE39 训练尚未启动。

执行结果以[完成回执](../results/gftd_observability_20261107/execution_receipt.json)为准；started.json 是不可重放的启动记录，其中 RUNNING 不是当前进程状态。汇总见[本轮推进报告](../reports/gftd_progress_20261107.md)。

旧状态原文保存在 [恢复前状态](history/current_state_before_gftd_resume_20261107.md)。原[结论](../reports/p_constant_norm_20261031_conclusion.md)、[科学判读](../reports/p_constant_norm_20261031_scientific_assessment.md)和[最终独立审查](../reports/p_constant_norm_20261031_final_review.md)保留不改。
