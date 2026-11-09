# Run 协议：closure_probe_20261030（残差归属探针，v2）

状态：**FROZEN（root 采纳，独立挑战已回应）**；日期 2026-10-30。
授权语境：用户 2026-10-30 指令“按项目规划开始推进”；本轮**只做已授权数据上的无训练分析**：
复用 train fit297 的冻结预测/历史/模型，不恢复训练、不开 val/test、不新增 FOM、不移动旧证据。
Campaign：[force_memory_closure_20261030](../campaigns/force_memory_closure_20261030.md)。
架构简报 v2（本协议引用）：[architecture_brief_20261030.md](architecture_brief_20261030.md)，
SHA256 `fed1ddb5de5d9b11a453da3808b36ac8a717c2ced39194182fd24c11ded0a8b7`。
独立挑战：[architecture_brief_20261030_challenge.md](architecture_brief_20261030_challenge.md)。
父候选：A96→A128（source `181d7f8bba2eb530a0ad5afe0130035755e974a2`）；保护参照 R_N；source commit `61f8442`。

目的：**probe**。关键问题：A128 与 R_N 的稳定 5/8 s 差距、以及失稳真实区间的相反权衡，
主要落在共同模态闭合、初始隐力可辨识性、优化/选择、还是冻结机械体/端口误差；
本 probe 要缩小解释集合，或在证据不足时明确写“未决”。
负结果会改变哪个判断：若标量闭合残差被 W1 差异解释、且 r₀ 差异在后验散布内，则 E 扩展的动机减弱；
若 r₀ 差异超后验散布，则先修初始化，不先加结构。

实际模型回执：设计来源为既有 Astra 条件准备（`astra6_next_structure_preparation_20261029.md`，
实际 gpt-6-astra，2026-10-29）；本 probe 的修订、阈值与判读由 root（本会话 actual deepseek-flash）完成，
不冒充 Astra 对本 probe 的逐条背书；独立挑战已由未参与写作的 reviewer 完成。

## 冻结资产（只读）

- 事件与真值：SGM `checked_preparation(private_data/state_gated_memory_20261028/study01, fit_only=True)`，
  297 条、30 失稳；train fit297 已消费，不是独立确认。
- 预测：`private_data/state_gated_memory_20261028/study01/qualification/R_N_fit297/predictions_*.npz`；
  `private_data/memory_transport_20261029/study01_prelaunch_retry1/study/{A,S,P}_30311/fit000128/predictions_*.npz`。
- 模型：A128 `.../study/A_30311/checkpoints/step000128/model.json`（`mass/damping/jpi_p/storage_base/p_scale`、
  κ、λ）；S/P/A96 仅用于 P3 的初始化器比较；R_N 的机械 buffer 若不同则分别使用并在结果中写明。
- 指标：`train_structured_rom._point_error`（四项）、`METRIC`、`_coi`；聚合 `train_local_memory_0098.risk`。

## 测量

**P1 描述性分解**：R_N/A128/S128/P128 的四项风险（稳定 5/8 s、失稳真实区间）与 A−R_N 差距分解。
官方点风险路径：`fine[::2][1:steps+1,:30]` 对 `x[11:11+steps]`（steps=500/800）；
失稳：`n=min(1500,len(x)−11)`，`fine[::2][1:n+1]`；coarse 作独立交叉值。**只作描述**。

**P2 机制主检验（COI 预算 + 标量闭合）**：按架构简报 v2 的公式计算
`r_est = 机械反演（冻结 M、D、q_p；ω̇_true 中心差分）`、`R1_est`、`Δω_COI`，
以及 `I_r, I_D, I_p` 三项积分份额与预算闭合残差；用模型 κ、λ 与 `W1_true` 驱动标量闭合，
比较 `R1_closure` 与 `R1_est`，并做逐事件最佳 (κ,λ,R1(0)) 拟合。区分两种残差来源：
闭合形式不足 vs `W1_true−W1_model` 差异。A128 与 R_N 都做；S/P 若成本允许一并。

**P3 初始化**：`history_system` 的 50×10 设计的奇异谱/条件数/ridge 贡献（仅条件数与先验敏感度）；
A128/S128/P128/A96 四个初始化器在同一 297 条历史上的 r₀ 差异（`‖Δr₀‖/‖r₀‖`、`Δ(1ᵀr₀)/10`），
以及相对拟合残差后验散布的倍数。**不把 ridge 主导解释为数据不可辨识。**

**P4（条件执行）**：冻结 64 步 checkpoint 在 full fit297 上的 5/8 s 风险，检验 panel48 的 64→128
回升是否在全队列复现。仅在 P1–P3 完成后决定是否执行；执行则记录为同一 Run 的第二个 Job。

## 验收门

1. 八个公开 5/8 s 风险 + 三个失稳风险复现到 <1e−9（fine 路径；A/S/P/R_N）。
2. common ω RMSE 8s 与 normalized p RMSE 8s 复现（A/R_N 至少，容差 1e−9）；分母 267/30、unresolved=0。
3. 逐事件四项和 == `_point_error`（<1e−12）。
4. P2 预算闭合残差仅达有限差分/离散量级；标量积分器在人工已知 (κ,λ) 轨迹上可恢复；
   `r_est` 的符号/索引对齐有单测。
5. 不通过 → 只报告实现无效，不做科学判读。

## 判读规则（预登记，比较尺度而非固定百分比）

1. **H1**：若标量闭合残差在 `W1_true−W1_model` 差异之外仍显著（相对该差异的放大倍数报告，
   不预设阈值），且 COI 预算中 Δr 项占主导 → H1 进入下一判别（先做零训练的结构可达性检验，
   再决定是否登记 E 的最小结构 Run）。若闭合残差主要被 W1 差异解释 → H1 降级。
2. **H2**：若模型间 r₀ 差异或先验敏感度超过拟合残差后验散布（报告倍数）→ H2 存活，先修初始化；
   否则 H2 不作主要解释。
3. **H3**：P4 复现全队列 64→128 回升 → H3 升级；未复现 → H3 主要证据削弱。
4. **H1 与 H2 同时存活**：先修初始化（更便宜，且 r₀ 偏差会污染 H1 的解释）。
5. 都不成立：写“未决”；不自动进入 H3 适配，不直接采用 E 或 s。

## 有限顺序、成本与停止

- Job 1：P1+P2+P3（CPU 分析；记录 wall/峰值内存；optimizer 调用 0）。
- Job 2（条件）：P4 单模型 full fit297 前向评估（无训练；记录 wall；不改契约）。
- 实现修复至多一次原套件重跑；数值门失败不挑事件/幅值/步长；旧 `memory_transport` 余额不转。
- 关闭：三个独立决定（代码/模型/科学）；模型不晋升，default none 不变；证据按 §归档。

## 强主张条件

本 probe 不申请性能/机制强主张；只报告“在此冻结证据、四项定义与冻结机械体下，误差落在哪、
r₀ 由模型/数据决定多少、标量闭合能否解释共同模态”。不把共同项大直接称结构闭合不足；
不把 ridge 主导称不可辨识；不把一次分解称确认。证据路径：
`private_data/closure_probe_20261030/p1.json`、`private_data/coi_budget_20261030/budget.json`、
`private_data/closure_probe_20261030/p3.json`、`results/closure_probe_20261030/public_probe.json`、
`results/coi_budget_20261030/public_budget.json`（含 SHA256）与 harness run 记录/`assess` 输出。
