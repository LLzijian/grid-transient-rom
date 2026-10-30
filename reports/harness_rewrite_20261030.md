# Harness 重写记录（2026-10-30）

用户要求：AGENTS.md 保持简洁；任务设计移入 kiki.md；区分"改进无效"与"优化无效"，
一轮不能判定、需要更详细的机制验证；防止局部工程优化压倒核心方向；第一性原理与
简洁/优美的模型审美保留在最高层（AGENTS.md）；参考 TASK4 的 harness 设计，只读。

## 改了什么

| 文件 | 作用 | 规模 |
|---|---|---|
| AGENTS.md | 顶层：使命与审美、核心方向纪律、Campaign/Run 组织与三决定、机制验证总则、文件职责、角色、授权硬边界 | 118 行 / 3.9K 字符（原 180 行 / 13.8K 字符） |
| kiki.md | 任务设计：物理对象与部署合同、数据边界、模型设计空间与硬约束、L1/L2/L3 指标、机制验证设计（L0–L4 + 机制卡 + 判定表）、当前方向与 Campaign 示例 | 272 行 / 8.2K 字符 |
| docs/research_policy.md | 证据三标准、双档案与四指针、数据边界、预算与账目、Git 归档、Campaign/Run 最小发布包与状态 | 120 行 / 3.7K 字符 |
| docs/engineering_loop.md | 流程：Campaign/Run/Job 实验树、主流程、优化充分性、机制验证、快速迭代边界、诊断链与工程修复、停止/延长、方向纪律、三个独立决定、监控、Astra6/挑战/归档 | 193 行 / 6.4K 字符（原 327 行 / 24.9K 字符） |
| templates/ | campaign / run_protocol / run_closure / mechanism_card 模板 | 142 行 / 3.3K 字符 |
| README.md | 更新入口导航 | 55 行 / 3.2K 字符 |
| docs/history/ | 原 AGENTS、原 engineering_loop、当前 current_state 快照 | 只读档案 |

规则总量约 22.2K 字符（4 个规则文件），低于原 38.7K 字符（约 −43%）；AGENTS.md 单独下降约 72%。

## 关键设计（参考 TASK4 新 harness）

1. 分层职责：AGENTS（原则/权限/导航）— kiki.md（科学任务/部署/评价/机制验证）—
   research_policy（证据/档案/预算/Git）— engineering_loop（流程）— 状态与轮材料。
   规则文件禁止记录实验流水或成绩。
2. 机制无效 vs 优化无效：三命题分离（优化充分性 / 机制实现探针 / 能力归因）；
   单轮只允许"优化未决/机制未实现/范围内未观察到收益/范围内支持"；机制否定须
   优化充分 + 探针通过 + 匹配对照 + 重复确认；资源不足记 unresolved_within_budget。
3. 防止局部工程：每轮标注科学层级；工程只作为科学轮次的使能工作；连续两轮只剩
   调参/QA/记账即强制结构复盘并暂停用户；campaign 闭合做三行方向审计。
4. 顶层审美：第一性原理分解、最简有效模型、复杂度必须付费（消融）、可分析/可辨识/
   可迁移、命名与主张诚实。
5. 证据结构沿用 TASK4：三曲线（能力/残差/成本）不合成总分；三独立决定（代码/模型/
   假设）；双档案（可用模型保护 vs 科学候选）；范围化结论与历史不重判。

## 未改什么

- 科学任务、部署合同、数据边界、预算类别与用户授权不变；未启动或修改任何实验、
  结果、模型指针与当前科学结论。
- docs/current_state.md 未改（仍为上一轮暂停时的旧快照，待收尾更新）。
- TASK4 只读：未写入、未运行、未做任何修改。

## 验证

- 全部新/改文件 Markdown 相对链接解析通过（模板占位符除外）。
- 原文件已存 docs/history/；新规则支持原有命名、状态词汇与归档仓库。
- 尚未 Git 提交/推送；与上一轮待收尾的归档一并处理。


## 追加：Campaign / Run / Job 与三个决定（2026-10-30）

用户进一步要求借鉴 TASK4 的实验树设计并显式区分三个决定。已更新：

- AGENTS.md §2：Campaign→Run→Job 组织；一次 Run 未超过可用模型不等于 Campaign
  失败；每 Run 分别记录代码/模型/科学三个决定，不得用一个 accept 包办；回退活动
  代码或指针，不回退研究记忆。
- docs/engineering_loop.md §1：实验树（Campaign 假设卡、Run 目的类型 probe/
  adaptation/discrimination/confirmation、Job 恢复语义、实验树管理者）；§5 快速
  迭代（借鉴 autoresearch，但禁止固定短时间/单指标贪心决定结构假设）；§8 三决定
  表与允许组合；其余章节改用 Run 语义。
- docs/research_policy.md：Campaign 卡+Run 树的归档、三决定的分别记录、三级状态词。
- campaigns/README.md 与 templates/campaign.md：Campaign 假设卡与 Run 树档案。
- templates/run_protocol.md、templates/run_closure.md：替代原 round_* 模板，加入
  Campaign/父 Run/目的类型、快速迭代边界与三决定表。
- 删除本轮新建的 round_protocol.md / round_closure.md（未归档的新文件，非历史证据）。
- Run 术语统一；历史 reports/results 路径与身份不变。
