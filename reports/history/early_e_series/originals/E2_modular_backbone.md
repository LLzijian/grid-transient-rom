# E2 技术报告(阶段 1):能量结构化主干 + 候选模块提议

> 历史保留标记（2026-09-22）：本报告保留为早期开发/候选模块提议记录。
> 当前状态见 [current_state](../docs/current_state.md)；结论边界以 [0010 审计](accepted/0010_project_conclusion_audit.md) 为准。

日期: 2026-09-04 · 状态: **PASS(backbone + proposal)** · 证据:
`results/e02_backbone/{summary.json, proposal.json, model_base.npz}`
(F1,generator_10,ZOH;val 43 事件;test 未打开)

## 1. 目标与协议

在冻结基座(不可删除的 M>0, D≥0, K≥0 摆主干)上,先用**无模块主干**
拟合,再从**积分残差**(无数值微分)提议支持度最高的物理模块候选——这是
"模块化 SINDy"的第一层:模块不是穷举,而是由残差的时标/振荡特征**数据驱动
提议**,随后由 2% 非劣门做整块取舍。

调用链(与 kron-reduce `data_level_structure_selection` 一致,映射到 5 折
train/val):`PrunableEnergyModel(active_blocks=()).fit(train,val)` →
`propose_candidates(train, D, K)`。

## 2. 主干拟合结果

| 量 | 值 |
|---|---|
| 求解器 | equivalent normal-equation Cholesky **非负**最小二乘 |
| 标量方程 | 8,178,400(600 等权块:source→fault-line→event→window) |
| 参数 | 55(10 D + 45 K) |
| 有效秩 / 条件数 | 55/55 · 18.9 |
| **弱残差 RMSE** | **2.73e-5**(比 M0 同口径更低,主干内部自洽) |
| 可辨识门 | pass |
| 物理审计 | pass:非负、互逆动态端口、能量供给恒等式按构造成立 |
| val H100 mean RMSE | 0.381(≈ M0 的 0.374,纯主干无增益——预期,模块未装) |
| 失败率 | 0 |

## 3. 模块候选提议(核心科学产出)

残差积分冲量在 209 个训练事件上做 **source 内 fault-line bootstrap**(200 抽),
支持度阈 0.8:

| 模块族 | 支持度(高/密双源) | 是否入选 |
|---|---|---|
| NONLINEAR_POTENTIAL | 0.0 / 0.0 | ✗(无独立残差支持) |
| EDGE_DAMPING | 0.0 / 0.0 | ✗(无独立残差支持) |
| **NODE_HIGH_ORDER** | 1.0 / 1.0 | ✅ τ=0.2 s |
| **EDGE_MEMORY** | 1.0 / 1.0 | ✅ τ = 0.02 / 0.22 / 1.62 s |
| **RESONANT_MEMORY** | 1.0 / 1.0 | ✅ (0.20 Hz,ζ0.49)、(0.80 Hz,ζ0.1)、(1.70 Hz,ζ0.1) |

- **与旧项目 clustered_7 AFSD v2 选出的 SS-DM 路线呼应**:记忆类模块
  (memory/resonant/node-lag)在两套表示里都是主力。
- **差异(值得写进论文)**:10 机 + ZOH 下非线性势与边阻尼不再被残差支持
  ——说明纯 M0 同步力矩已吸收准静态势,余留主要是**多时标记忆与低模
  振荡分量**;低模 0.20 Hz / 0.80 Hz 对应区间振荡。
- `median_residual_ratio = 1.018`:装模块前平均残差仅比输入噪声地板高
  1.8%——**基座已经把 98% 以上的动力学解释了**,模块面对的是很薄的
  残差。这是论文"基座 + 精确补差"叙事的量化起点。

## 4. 结论与下一步

主干与候选提议健康。`EDGE_MEMORY×3 + RESONANT_MEMORY×2 + NODE_HIGH_ORDER`
6 实例进入全模型联合弱形式拟合(fit 阶段),随后做整块后向剪枝。
