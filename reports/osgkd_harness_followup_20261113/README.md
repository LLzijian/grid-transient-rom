# OSGKD 入口、设计归档与角色回执补充

本次完善三项：活动设计入口指向 OSGKD，11 份旧 GFTD 设计按原字节归档，阶段一的 Scientist、Coordinator、Reviewer 调用补入可查宿主证据。阶段二未释放，本次不启动训练。

## 入口与归档

[设计入口](../../designs/README.md)直接连接 [OSGKD 架构](../../designs/osgkd_architecture.md)和 [Campaign](../../campaigns/osgkd_ieee39/campaign.md)。[当前设计](../../docs/current_design.md)中的前序设计链接指向 [GFTD 历史档案](../../designs/history/README.md#gftd-20261113)。原有部署合同、研究设计和暂停状态保持原范围。

11 份归档的路径、字节数、SHA256 与原提交阅读链接见 [迁移清单](../../results/repository_organization/20261113_gftd_design_archive/migration.json)。现有组织入口增加一个经过哈希固定的设计归档输入；旧协议和旧迁移清单不改写，导航先核验归档字节，再按旧记录的字节数和哈希核对。

原 Markdown 的相对链接保持历史语义，GitHub 不读取本地归档映射；历史索引另提供固定提交的阅读入口。没有在活动目录新增旧设计占位文件。

## 原阶段一角色调用

[宿主调用摘录](../../results/osgkd_harness_followup_20261113/role_receipts.json)保存父聊天、子线程、原 spawn 配置、阶段一 followup 调用、子轮次开始/完成时间及最终答复编号。

| 角色 | 已核实的宿主配置 | 阶段一完成消息所对应的职责 |
|---|---|---|
| Scientist | gpt-6-astra | 起草 OSGKD 架构和 Campaign，提出集中审查重点 |
| Coordinator | gpt-6.1-sol | 更新 11 份控制文件，切换活动合同及导航 |
| Reviewer | gpt-6.1-sol | 审查阶段一材料，核对 19 项清单并完成审查报告 |

三者原 spawn 均为 `fork_turns=none`，阶段一通过 followup 复用已有子线程，不能描述成阶段一新建独立上下文。宿主配置和调用完成可核实，`server_resolved_model` 保持 null。角色分工和审查完成不代表独立科学复验。Reviewer 阶段一的九条可见命令中，两次写入均为审查报告，另七次为读取、哈希或差异检查；加密消息和完整先前参与关系仍为 unknown。

Scientist 完成时的架构哈希与最终架构一致；当时的 Campaign 哈希与最终采用版不同，两个版本分别记录。Reviewer 最终消息中的报告哈希与 [冻结审查报告](../osgkd_transition_20261113/review.md)一致。

## 冻结材料与本次变更

原阶段一 `adoption.json`、`closure.json`、`review_inputs.json`、`validation.json`、审查报告、来源快照，以及旧 Run 登记均保留。原阶段一来源提交为 `0ff2c9643d9eae11ca00620ed816e58beeba5dc9`。

本次只更新活动 `docs/current_design.md` 中的旧设计链接；它在原清单对应的字节另存于 [更新前原件](before/current_design.md)，与原清单 SHA256 一致。旧清单继续描述原提交，不能拿它当本次导航修订的清单。

本次控制面测试为 **91 项通过、209 个子项通过**，覆盖旧记录导航、路径与字节校验、调度及证据判读。验证明细见 [核验回执](../../results/osgkd_harness_followup_20261113/validation.json)；本次独立复核见 [审查回执](review.md)。未运行模型训练或科学资格实验。

本次审查清单同时记录原始字节哈希和 LF 规范化哈希：前者固定审查时工作副本，后者用于普通文本跨平台检出比对。冻结记录和归档原文继续以原始字节为准，不能用换行规范化替代原校验。
