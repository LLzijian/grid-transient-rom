# Run 闭合：<id>

一句话：<在何范围观察/支持什么，什么仍未决>。
来源 commit、协议/修订、checkpoint/hash、数据/尺度/seed、实际成本与未知项。
执行状态；预算状态；科学状态分别写，绝不合成一个 accept。

| 证据轴 | 判断 | 可审查引用与限制 |
|---|---|---|
| 有效性 | valid / invalid / unknown | |
| 优化健康 | healthy / unhealthy / unknown / NA | |
| 优化充分 | supported / insufficient / unknown / NA | |
| 机制探针 | passed / failed / unidentifiable / NA | |
| 效应/确认 | 正/负/权衡/等效/未决；确认范围 | |

关键指标含原分母/时域、逐事件身份、风险与 RMSE 分别报告；记录负面结果和成本。
实际发生的偏离及解释；旧记录不回写。strong claim 的材料只在申请时附。

三个决定：code=<...>；model=<...>；science=<...>，各自理由与范围。
confirmation_need 与 promotion_eligibility 分开，工具记录资格不代替 root 判断。
新增知识；仍在竞争的解释；下一最有判别力的问题或暂存/重开条件。
主题维护：受影响的 Campaign 问题／正文位置；直接核对的原结论与证据；认识变化与更正范围；尚未纳入材料。
材料导航：Run 入口、原执行／技术续跑关系、结果与本地产物清单；新增路径保护项及解除条件。
归档 commit / push 核验 / 未解决阻碍；真实清理与下一授权动作。
每个闭合 Run 交付一个 `artifacts/research_records/<run_id>_records.zip`。本轮登记、
协议、结果、审查及复现源码入包，执行回执和逐事件记录用 `--local-record` 指定。
本轮新增、位于通用目录的设计或源码可用 `records_archive_source_refs` 指定，由执行释放中的来源哈希校验。
完成本地归档提交后运行 `python tools/archive_run_records.py <run_id>`，逐项核验
包内清单和 SHA256。原始轨迹、私有数据、模型权重和优化器状态按本地资产清单保全。
