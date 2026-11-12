# GFTD 收束：保留可复用成果，暂存未完成支线

2026-11-13，按用户本次研究转向收束。GFTD 的后续投入暂存，OSGKD 接替活动主线；这是优先级决定，已有正面、负面和未决结论各保留原范围。

| 已有事实与范围 | 对新方案的意义与来源 |
|---|---|
| 四个无噪 K4 条件恢复静态支撑；不外推任意图 | 继承候选图、弱式消元和绝对信息守卫。[静态恢复](../../results/gftd_sparse_topology_20261107/summary.json) |
| 原无界指数族存在储能沿幅值方向无下界的条件，并有有限时逃逸反例；有界核及完整梯度已有修订证据 | 保留数学反例、核及数值实现；新增机械/环境动力学不能自动继承其存在性证明。[储能分析](../gftd_storage_feasibility_20261109/conclusion.md) |
| 最新 train fit297 匹配预测两模型均为 297/297 初始化原六门通过；调制严格 1 s 参考为 289/297，其余 8 到限 | 不用原 Run 3 的 46/297、1/297 代替最新状态；初始化与长时预测资格分开。[匹配预测](../gftd_matched_prediction_20261109/final_assessment.md) |
| 真实区间失稳漏检为静态 21/30、调制 17/30；完整可比 risk 未支持调制总体收益 | 多检出 4 例；静态三个误报在调制侧对应数值未决，不能称误报修复。两模型实际初态 L2 各 0/297 获证。[最终决定](../../results/gftd_matched_prediction_20261109/final_adoption.json) |
| 原 24 包历史选择目标改善 0.135032%，10 包 MSE 降、14 包升 | 它是历史拟合结果，未产生该候选未来预测成绩。[三点判读](../gftd_objective_alignment_20261112/final_assessment.md) |
| 两档固定局部方向的观测变化范数消减约 96.8%，剩余归一化 RMS 2.304e-5、1.153e-5 超过 1e-6；求解到限 | 见证未决；不改写为预测改善或不可辨识性证明。[判读](../gftd_joint_distinguishability_20261112/final_assessment.md)及[计时更正](../gftd_joint_distinguishability_20261112/final_assessment_correction.md) |

旧 GFTD 候选保留原 p 端机械动态与映射、幅值/力记忆假设和实际训练范围；部署采用原 100 ms 相角、转速与真实末点 p₀。它们作为 `known_p0_reference` 类信息增强参照，不能直接赋予新的 `motion_only_v1` 合同，旧 r 也不能仅改名为环境状态。

最新线性界子问题的设计已采用，两版脚本均已保存，第二版实现复查因暂停中止。资格检查、精确证书和 BVLS 科学求解均未执行；未用工作量现已关闭，不转入 OSGKD。保留原 `PAUSED_BY_USER` 执行状态并标注暂存去向，科学结论为 `unresolved`，不伪造 COMPLETED。仓库无实际派发器；示例任务不是活动队列，旧任务的下一动作改为 `defer`、自动派发关闭。

代码 `retain_branch`、模型 `retain_evidence`、科学 `unresolved`。源码、模型、原结果和冻结协议保留，default none、incumbent 与 protected anchors 不变。[状态与字节证据](../../results/osgkd_transition_20261113/closure.json)
