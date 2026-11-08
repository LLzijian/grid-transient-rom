# 闭合方程发现：结构搜索、因果初始化与工程核验

**这组历史研究建立了可复查的闭合方程搜索和导出流程，并把结构选择、初始化与执行实现的问题逐项分开。阅读时先区分：哪些结果支持限定的科学认识，哪些只是工程通过，哪些尚未得到有效比较。**

本页于 2026-11-08 追溯整理，覆盖 2026-10-17 的 A—E 研究及 10-19 的机制与工程记录。原件、当时的采用决定和数据暴露历史保持不变；当前授权与候选见 [kiki.md](../../../kiki.md) 和[当前研究状态](../../../docs/current_state.md)。

## 按问题阅读

| 问题 | 当时形成的认识 | 主要记录 |
|---|---|---|
| 生成式搜索是否带来固定库之外的收益？ | B 阶段两种搜索选中相同主结构；C 阶段相同种子的主候选与固定库模型字节相同。完成搜索与重训，不自动证明生成式方法更优。 | [B：结构选择](originals/discovery_20261017/closure_discovery_20261017_stage_B.md)、[C：重训与删除比较](originals/discovery_20261017/closure_discovery_20261017_stage_C.md) |
| 新条件上的确认是否支持整体改善？ | 原报告的可评价稳定子集有小幅 5 s 风险改善，同时有失稳漏检损害；原登记的完整确认保持未决，全部端点仍为研究候选。风险与 RMSE 分开解释。 | [A—E 最终报告](../../closure_discovery_20261017_final.md)、[D：原确认记录](../../closure_discovery_20261017_stage_D.md) |
| 初态信息与方程表达应如何区分？ | 已知支撑的 18 个控制均未达到登记预测门槛；额外给定真实 q 初态的诊断支持进一步检查初始化。真实 q 属于额外信息，不能作为部署成绩。 | [P0／P1 进展与控制结果](originals/mechanism_20261019/closure_mechanism_20261019_progress.md)、[真实 q 诊断](originals/mechanism_20261019/oracle_q0_20261019_conclusion.md) |
| 100 ms 历史是否能帮助因果初始化？ | 在给定节点机制的合成问题上，历史残差初始化的三个种子达到登记标准，对照均未达到；该结论针对整体初始化改动与已知支撑，不等于真实电网机制恢复。 | [节点配对结果](originals/mechanism_20261019/node_causal_20261019_conclusion.md)、[边机制原结果](../../edge_causal_20261019_conclusion.md) |
| 饱和机制没有得到成绩，是什么原因？ | 首次启动在更新前因身份检查失败；后续修复路线又在不可变缓冲区检查处关闭。没有可评价的六项科学比较，不能据此否定初始化或闭合方程。 | [首次失败](originals/mechanism_20261019/saturation_admission_failure_20261019.md)、[路线关闭记录](originals/mechanism_20261019/saturation_pair_20261019_conclusion.md) |

## 工程接受与科学采用分别阅读

| 工作 | 原记录中的决定与适用范围 | 来源 |
|---|---|---|
| JSON 身份保存与恢复 | 接受新增工程接口；合成 CUDA 检查覆盖参数、Adam、随机状态和恢复后下一更新。 | [工程结论](originals/mechanism_20261019/checkpoint_identity_20261019_conclusion.md) |
| 饱和端口专用计算 | 固定非空工程网络有正确性和提速证据；通用接口集成暂缓，空边集的梯度语义问题单独保留。 | [原结论](originals/mechanism_20261019/tanh_port_runtime_20261019_conclusion.md) |
| 构造顺序与固定边项费用核验 | 前者检查中断、未进入测时；后者完整恢复与费用核验未完成。两者均不作为机制作用的科学比较。 | [构造顺序](originals/mechanism_20261019/construction_order_20261019_conclusion.md)、[固定边项核验](originals/mechanism_20261019/edge_pair_gate_20261019_conclusion.md) |
| GPU 训练入口 | 原协议限定工程实施，具体结果与适用条件沿原报告核查；归档协议不恢复旧研究。 | [原工程协议](originals/discovery_20261017/gpu_training_20261017_protocol.md)、[结果报告](../../gpu_training_20261017.md) |

原模型咨询与执行方案随结果保留在同一主题中，用于回溯当时的决定。历史预算、命令和自动推进描述是原件内容，不作为今天启动任务的指令。

## 查原件与路径依赖

本主题累计收拢 24 份原件；[材料表](catalog.md)同时提供已归档原件和仍在原位的相关报告，保留原标题、材料用途及固定提交阅读入口。[保护入口](protected.md)连接最新的引用者、绑定类型和解除条件。

原件字节保持不变。跟随原件内部链接时，使用材料表的固定提交版本；部分本地原件与 Git 版本仅换行不同。新导航的本地链接单独核验，原件内部链接不回写。

- [联合训练与初始化的前期研究](../private_rom/README.md)
- [本批迁移与累计位置](../../../results/repository_organization/20261108_early_history04/migration.json)
- [整理批次总表](../../../results/repository_organization/README.md)

10-19已知支撑对照的[旧批入口](../../../experiments/history/closure_discovery/README.md)已原样归档；完整旧版本保留同目录子脚本关系。

第八批在前序代码／概览归档后复核保护边界，再解除6份报告／证据的旧路径依赖；[最新累计位置](../../../results/repository_organization/20261109_history08/migration.json)与本主题保护表接续原记录。
