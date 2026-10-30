# Run 闭合 <run_id>

Campaign：<campaign_id>；父 Run/候选：<id>；目的类型：<probe/adaptation/
discrimination/confirmation>
一句话结论：<范围内可被证据支持或否证的结论>

## 身份与范围
来源：<commit/checkpoint>；数据与分母：<范围>；评价：<时域/指标/数值资格>
归档：<branch/commit>；推送核验：<remote sha 或 publish_pending>

## 研究了什么
<检验 Campaign 的哪个关键部分；假设与已有方法的差别；不写工作量代替问题>

## 真正发生了什么
<与匹配对照/起点/强基线的比较；覆盖、关键事件、复杂度与成本；三曲线分开>

## 最重要的失败
<误报/漏检身份、最坏轨迹、未覆盖、数值未决、缺失证据>

## 如何解释
<已证实 / 合理推断 / 不能区分；优化是否充分；机制探针是否通过；
  是否达到机制否定所需的确认条件；剩余替代解释>

## 三个独立决定（不得用一个 accept 包办）
| 决定 | 记录 | 选项 |
|---|---|---|
| 代码决定 |  | merge / retain_branch / revert_active / no_change |
| 模型决定 |  | promote_incumbent / add_frontier / retain_evidence / ineligible |
| 科学决定 |  | continue / supported_within_scope / mechanism_null_within_scope / falsified_within_scope / unresolved / abandon |

允许组合：代码合并而模型不晋升、假设继续；方法无收益、代码不进主干，但完整结果与
负面认识仍提交推送。回退的是活动代码或工作指针，不是研究记忆。

## Campaign 更新
<在 campaigns/<campaign_id>/campaign.md 的 Run 树追加本 Run；更新信念、替代解释、
下一最有判别力的 Run；方向审计三行>

附件：配置、逐事件数字、曲线/残差、方程、成本、源码与复核记录。
