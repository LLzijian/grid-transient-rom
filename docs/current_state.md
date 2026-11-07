# 当前研究状态

**2026-11-07：完成“发现失败—末点估计改进—新方向确认”循环。完整初始化在 128 个新含噪输入上全部达标，已获独立复核，登记为限定人工条件下的工作方案。当前无运行中的科学 action。**

## 当前范围与指针

- 原始 val/test 关闭；旧消费 test 永久退出选择；新 FOM=0；TASK4 暂停。
- 部署输入为 11 点、100 ms 因果 theta/omega 历史及末点真实 p，之后共享自治 ODE。
- default none；incumbent 与 protected anchors 保留原适用范围。
- 当前工作初始化为 simple600→smooth1200：先从零 r 起点拟合，再联合估计末点相角、转速和 r。末点 p 硬约束、a_end=0、s=0.5 方程与已知共享参数固定。只适用于登记人工三机四状态、两档噪声、100 ms 历史与 1 s 预测条件；第一阶段保留为比较基线。
- 用户已授权按算法设计自主研究、测试与改进；范围内新 Run 经前瞻设计、独立审查及 root 释放，不逐轮请求用户确认。

## 最新确认及决定

[gftd_endpoint_confirmation_20261107](../reports/gftd_endpoint_confirmation_20261107_conclusion.md)使用 64 个新噪声方向，每方向配对 sigma=1e-6 与 1e-4。第一阶段为 125/128 达标，完整流程 128/128；另四条无噪基准复用且全部达标。1 s 未来相角/转速联合尺度化 RMS（尺度 0.1，人工无量纲）的要求为不超过 0.001，峰值不超过 0.01。

高噪 64 输入的平均 RMS 从 0.000542279 降到 0.000441290，下降 18.623%；全部新输入最坏 RMS 从 0.001107038 降到 0.000880095。104 个输入误差降低、24 个增加，增加者仍达标。每个“状态×噪声”层为 16/16，成功率 95% 双侧区间 [0.794093, 1]；不把两档配对视为独立重复，也不作高可靠覆盖保证。

256 个优化任务、29746 次残差调用、424.543 s，0 细步重拟合、0 执行后修复、0 重放。完整流程残差调用约为第一阶段的 7.12 倍。全部数值检查通过；第二阶段 24 个任务触边，5 个所选点的 KKT 未知，优化充分性保持 unknown。[科学判读](../reports/gftd_endpoint_confirmation_20261107_scientific_assessment.md)、[独立结果复核](../reports/gftd_endpoint_confirmation_20261107_result_review.md)、[root 决定](../results/gftd_endpoint_confirmation_20261107/root_adoption.json)与[执行回执](../results/gftd_endpoint_confirmation_20261107/execution_receipt.json)分别保存。

代码 retain_branch；模型 retain_evidence，并采用上述实验内工作初始化；科学完成有限新方向确认，Campaign 继续。该 Run 有限工作量已关闭，余额不转移。通用 GFTD 模型与既有受保护模型均未晋升或替换。started.json 是历史启动凭据，不代表当前仍运行。

## 前序证据与下一问题

- [5N 修订](gftd_repair_design_20261107.md)：43 项原资格及两图 32 项补充检查保留；局部观测探针在三个非平衡调制条件呈现局部 7 秩、约百万条件数。
- [初始化探针](../reports/gftd_initialization_20261107_conclusion.md)：12 输入、36 优化任务，简单方案 12/12，联合 a/r 方案 8/12。提交 5e02f98a648848318fd9d32c2ce612c550bb2a62 已推送核验。
- [首次噪声确认](../reports/gftd_noise_confirmation_20261107_conclusion.md)：简单方案 127/128 含噪输入达标，加 4 条无噪为 131/132；input_072 为有效预测失败。提交 71fa9dfc46ac03f487df4c8abbf2f194733ba78f 已推送核验。
- [末点估计诊断](../reports/gftd_endpoint_denoising_20261107_conclusion.md)：在上述已暴露输入上候选 132/132、硬锚再拟合 131/132，高噪平均 RMS 降低 19.655%。四条对照同点损失分类经独立附录纠正，原结果保持不变。提交 ec8c8750c8804780ffd164ad63dc046ad762e25f 已推送核验。

下一关键问题为[共享耦合参数能否与逐事件记忆初态区分](../reports/gftd_structure_next_question_20261107.md)。先校准弱动量的 p/r 端口，再在关闭调制、固定边形状的最小问题中研究一个共享未知尺度；每个候选参数必须重新推断合法隐状态。这是有限架构准备，新 Run 尚未释放。

[结构接口盘点](../reports/gftd_structure_interface_20261107_audit.md)保留通用 fit、系数到部署核映射及非零调制通用初始化的缺口。未开展 IEEE39 训练、未知结构恢复或 L2 证书计算。旧常数支线及其预算保持关闭。
