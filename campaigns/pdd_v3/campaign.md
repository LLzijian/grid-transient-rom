# Campaign：pdd_v3（特权物理观测下的开放网络化动力学可辨识性与两阶段解耦发现）

- **登记性质**：`PROSPECTIVE`（2026-11-28 正式拍板启动；当前阶段仅限架构设计、只读核查与治理审查，**未放行代码实现或训练**）。
- **目标会议**：NeurIPS 2027（不硬赶 ICML 2027）。
- **当前科学状态**：`active`（规划审查中）；执行状态：`PAUSED_BY_USER`（完成架构简报、Scientist 有界设计、Reviewer 挑战并核实云盘同步后暂停待放行）；部署采用指针：`default none`。
- **架构版本**：[PDD v3 架构简报](../../designs/pdd_v3_architecture_brief.md)（`DRAFT`）；前代端到端参照：[OSGKD 2.0 架构](../../designs/osgkd_v2_architecture.md) 与 [v2 Campaign](../osgkd_v2/campaign.md)。
- **授权引用**：用户 2026-11-28 正式决策与授权——① 切换论文头条至 **LUPI for Open Networked Dynamics**（训练期特权端口 $\mathcal{O}_1$ vs. 部署期短时部分观测 $\mathcal{O}_0$）；② 确立“一主两辅”收紧范围（唯一主干瘦身版两阶段解耦算法 PDD + 借自方案 A 的 2–3 条核心可辨识定理 + 借自方案 C 的大样本合成失稳评测列）；③ **正式授权新建公开合成科学基准**，范围严格收紧为三项：扩大失稳样本量并显式构造强开放环境记忆/强调制工况、对齐 ML 顶会标配基线与标准指标、新增 1 个非电力且非纯振子领域的开放网络系统。
- **不可逾越红线**：新增高阶 FOM 仿真 = `0`；IEEE-39 `test` 严格零新增访问；不宣称跨领域/跨规模冻结权重零样本迁移；既有受保护历史锚点与 `default none` 不变。

**范围明确的假说**：在训练期具备特权聚合端口观测（$\mathcal{O}_1$）而部署期仅观测短时运动历史（$\mathcal{O}_0$）的开放网络化动力学中，只要显式限定环境函数类（奇偶分离 + 严格真因果记忆）并把静态互易骨架恢复（Stage 1）与固定骨架上的残差开放闭合及短历史部署初始化（Stage 2）解耦，即可在强开放/强调制工况下同时消除端到端模块互代、获得显著优于端到端 v2 的结构恢复，并显著优于 `Stage-1-Only` 与无特权同构基线，同时在大样本失稳评测中避免纯 MSE 黑箱模型的失稳抑制。

| 竞争解释 | 区分预测 | 已有证据 / 反例 | 尚未解决 |
| --- | --- | --- | --- |
| **Stage 2 冗余（同义反复质疑：只是 Weak SINDy）** | 若开放/调制较弱，`Stage-1-Only` 即可在长时 MSE、VPT 与失稳指标上逼近或超过 PDD | 旧 Layer 1 调制格弱式静态回归 4 s 风险已低至 `0.00207`（[WP2 报告](../../reports/osgkd_v2_wp2_layer1_20261127/result.md)） | 新基准强开放/强调制工况上的 G1 硬门禁（`PDD` vs `Stage-1-Only`） |
| **端到端联合拟合更优（Stage 1 静态偏移损害 Stage 2）** | 在强调制下 Stage 1 的静态刚度估计偏移可能限制 Stage 2，导致端到端 v2 长时误差更低 | 端到端 v2 在 Layer 1 静态边 F1 仅 `0.686`、调制 F1=`0`；任务 2 实测 `frozen_p` 端到端重拟合把 $p_0$ 压缩 3×（[val 核查](../../reports/pdd_v3_architecture_20261128/val_trajectory_audit.md)） | 新基准 G2 硬门禁（`PDD` vs `End-to-End v2` 及 `Stage-2 Bounded-Recal` 消融） |
| **特权端口并无实质增益（纯运动加结构先验即够）** | 同架构去掉 $\mathcal{O}_1$ 监督（`No-Privilege Same-Arch`）达到同等结构与预测表现 | Theorem 2 给出 $\mathcal{O}_0$ 下 $N$ 维端口平移规范与同频等价类 | 新基准 G3 硬门禁（`PDD` vs `No-Privilege Same-Arch`） |
| **黑箱连续模型在大样本下并不抑制失稳 / 直接分类器足以替代** | `Graph Neural ODE` / `Latent ODE` 在大样本合成失稳测试集上维持高召回，或 PDD 轨迹无法保住分岔 | IEEE-39 `val`/`test` 上 `Graph Neural ODE` 召回 `0/12`、`0/9`，任务 2 确认全负谱且 7/12 严格达稳；但留出失稳仅 9 条（统计效力不足） | 大样本合成失稳测试集 G4 门禁，并与 `Direct Classifier` / `Seq2Seq` 参照上界并列 |
| **仅限电力/正弦振子族，缺乏跨方程族普适性** | 在非振子开放网络（基准 3）上 PDD 失去相对 `Stage-1-Only` 或黑箱基线的优势 | WP3.1 摆阵（仍属二阶正弦振子）同模板重拟合优于 Neural ODE `90.3%`，但冻结迁移 `0/3` | 基准 3（Scientist 二选一：Wilson–Cowan 或黏弹性非线性弹簧-质点网）全套对照 |

## Run 树与前期准备

| 编号 / 阶段 | 父谱系 | 目的 | 冻结证据 | 代码 / 模型 / 科学决定 |
| --- | --- | --- | --- | --- |
| `pdd_v3_architecture_20261128`（筹备） | `osgkd_v2_frozen_confirmation_v2_20261128` | 战略定位修正、任务 2 零积分 `val` 核查、PDD v3 架构简报、Scientist 有界设计与 Reviewer 挑战 | [定位评估](../../reports/pdd_v3_architecture_20261128/positioning_assessment.md) · [val 核查](../../reports/pdd_v3_architecture_20261128/val_trajectory_audit.md) · [架构简报](../../designs/pdd_v3_architecture_brief.md) · [Scientist 设计](../../reports/pdd_v3_architecture_20261128/scientist_design.md) · [Reviewer 挑战](../../reports/pdd_v3_architecture_20261128/review.md) | `no_change` / `default none` / `continue`（待用户放行） |

## 已用适配与信息暴露

- **IEEE-39**：所有 `train`（385 条）与 `val`（117 条）均计入历史暴露；86 条一次性独立确认留出已于 2026-11-28 消耗封卷，**永久不再用于选择或再次盲测**；新 FOM 维持 `0`。IEEE-39 在本 Campaign 中仅作高阶旗舰应用与探索性参照。
- **新公开合成基准**：尚未生成、尚未训练、零暴露；首次执行前须完成 Scientist 有界设计、Reviewer 挑战与用户显式放行。

## 当前认识

OSGKD v2 端到端路线在 IEEE-39 一次性盲测上已明确：稳定轨迹精度逊于同编码器图 Neural ODE（范围内否证），跨领域冻结迁移失败，静态结构恢复逊于弱式稀疏回归；但机制必要性（删耦合/记忆均致失稳检出归零）、图初始化器在分布偏移下的防过拟合表现及可辨识负对照已获坚实证据。据此收束为 PDD v3（LUPI 两阶段解耦 + 3 条可辨识与适定性定理 + 大样本定性失稳列）。

## 下一个最有判别力的问题及最小工作

待本轮 Reviewer 挑战并经用户放行后，首轮最小判别工作（Run 0 资格与 G1 探针）应直接检验：**在基准 1（$N=8$）显式构造的强开放记忆 + 强调制工况上，Stage 1 能否在 $\mathcal{O}_1$ 下高 AUPRC/F1 恢复静态骨架，同时 `Stage-1-Only` 出现明显长时/失稳失配，而固定骨架的 Stage 2 PDD 显著跨过 G1 与 G2 门禁**。若在最小合成规模上无法打开 G1 空间，则暂缓全基线扩张并优先修正工况物理强度或闭合结构。

## 暂存 / 结束理由与重开条件

当前处于治理暂停态（等待用户审阅本轮架构简报、Scientist 设计与 Reviewer 挑战意见后放行）；不构成科学放弃。重开条件为用户明确批准进入代码实现与 Run 0 资格测试。
