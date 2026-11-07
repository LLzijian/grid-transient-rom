# 当前研究状态

**2026-11-07：用户已恢复研究并指示继续推进；GFTD 首轮合法初始化试验已完成。**

## 当前范围与指针

- 原始 val/test 关闭；旧消费 test 永久退出选择；新 FOM=0；TASK4 暂停。
- 部署仅用 11 点、100 ms 因果 theta/omega 历史与末点真实 p，之后共享自治 ODE。
- default none；incumbent 与 protected anchors 保留原适用范围。本轮没有真实事件数据读取或共享参数训练。
- 当前人工三机范围的工作初始化基线为末点 a=0、只拟合 r；仍使用 s=0.5 的完整幅值动力学。这不是关闭调制或晋升部署模型。

## 已完成证据

5N 方程修订及父模型映射见 [设计](gftd_repair_design_20261107.md)；43 项原资格与两图 32 项补充检查记录保留。前轮局部观测探针在三个非平衡调制条件呈现局部 7 秩、约百万条件数，结论仍限登记人工条件。

前轮提交 2454355298453769a15efffab5778cf5a2cfc222 已补推并核验远端一致；见[父归档对账](../results/gftd_initialization_20261107/parent_archive_reconciliation.json)。没有重跑旧科学 action。

本轮 gftd_initialization_20261107 完成 12 输入、36 个优化任务、10658 次残差积分调用，耗时 134.04 s；0 个细步重拟合。13 项接口与记账检查通过，正式计算和独立复核完成。候选只按合法历史选优，先固定拟合结果哈希再读取隔离真值。

在固定人工尺度及 1 s RMS/峰值要求下，简单初始化达标 12/12，联合 a/r 达标 8/12。24 个选定结果数值资格全通过，所有候选末点 p 保持真实读数，主细步平衡负校准均为零。联合的 24 个起点任务有 14 个用满预算；整体优化充分性未决。更低历史损失没有稳定转化为更好未来预测，不能据此否定幅值机制或声称状态唯一恢复。

完整结果见[结论](../reports/gftd_initialization_20261107_conclusion.md)、[科学判读](../reports/gftd_initialization_20261107_scientific_assessment.md)、[独立复核](../reports/gftd_initialization_20261107_result_review.md)与[执行回执](../results/gftd_initialization_20261107/execution_receipt.json)。started.json 是不可重放的启动凭据，其 RUNNING 字段不是当前进程状态。

## 决定与下一动作

代码 retain_branch；模型 retain_evidence；科学 continue。本 Run 工作量关闭，余额不转移。下一步先固定现有简单算法、误差尺度与预测门，用新的人工噪声方向确认覆盖；联合路线如继续，须先前瞻设计幅值收缩或复杂度约束，不能用本轮未来真值挑参数。下一 Run 尚未释放。

通用结构发现 fit 与筛选系数到部署核的映射尚未实现；已有固定三机、已知共享参数的初态推断原型。未开展 IEEE39 训练、未知结构恢复或 L2 证书计算。旧常数支线及其预算继续按原结论关闭；本轮不改变历史结论。
