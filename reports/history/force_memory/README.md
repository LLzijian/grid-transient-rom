# 力记忆与 p 通道：从空间作用到残差来源

**这组工作比较空间作用应该进入哪条方程，再逐项检查初始化、优化和机械功率通道的竞争解释。保留下来的知识包括限定范围内的正向效应、代价和被排除的解释；它们没有合成为一次最终模型晋升。**

本页于 2026-11-09 追溯整理。历史主体为 2026-10-25—31，承接[闭合模型与训练时域研究](../closure_development/README.md)，并成为[GFTD 主线](../../../campaigns/gftd_kuramoto_discovery_20261106.md)的前序证据。旧报告中的“当前”“暂停”和下一步属于原时点，今日指针见[当前状态](../../../docs/current_state.md)。

| 问题 | 已有认识与边界 | 直接来源 |
|---|---|---|
| 长期轨迹风险降低，是否同时改善关键事件？ | R-memory 冻结评价中，风险收益伴随部分范围的漏检增加；fit297、no_update88 和共同385分别报告。来源模型已参与 train385，不能把后者改称未见确认。 | [R-memory 恢复结论](../../r_memory_20261028_conclusion.md) |
| 空间作用进入静态耦合还是记忆传播？ | 在登记来源和有限适配下，传播方案比静态耦合方案有风险与8 s漏判收益，保留为后续研究起点；不等于真实线路恢复或最终采用。 | [结构比较](../../force_graph_20261028_conclusion.md) |
| 增加保守转运是否已解决失稳风险？ | 固定平面的共同128步比较未获登记收益，原 NOT_RETAINED 保留其不晋升含义；不升级为一般记忆族否证。 | [记忆转运](../../memory_transport_20261029_conclusion.md)、[独立方向挑战](../../force_structure_next_20261029_challenge.md) |
| 当时的主要缺口是初态、共同模态还是优化？ | 冻结探针不支持初始隐力可辨识性作为主要解释，共同模态解释在稳定队列被降级；检查点非单调加强了优化／选择解释。后续九点审计仍保留稳定精度与失稳识别的权衡。 | [残差探针](../../closure_probe_20261030_conclusion.md)、[检查点审计](../../checkpoint_selection_20261030_conclusion.md) |
| p 通道是否只有角速度误差传导？ | 来源诊断显示固定 p 方程残差与角速度误差传导均可分辨；反推数值门未通过时不作形式不足的采用判断，后续数值资格也不追认父判断。 | [来源诊断](../../p_port_diagnostic_20261031_conclusion.md)、[反推精度](../../p_reverse_precision_20261031_conclusion.md)、[解析反推资格](../../analytic_p_backward_qualification_20261031_conclusion.md) |
| 纯交换与共享常数补救分别排除了什么？ | 纯交换的精确实现否定对应条件数学命题；六机共享常数必要条件冲突只以已发布 binary64 范数为精确输入。两者均不直接识别真实驱动或否定所有近似模型。 | [条件实现命题](../../p_driver_order_20261031_conclusion.md)、[纯未来残差](../../p_residual_stage_20261031_conclusion.md)、[范数必要条件](../../p_constant_norm_20261031_conclusion.md) |

## 查材料与采用决定

- [材料表](catalog.md)按原始标题提供协议、结果、咨询和审查入口；每份原件保持一份，不补造历史 Job。
- [路径保护表](protected.md)解释为什么大部分正式协议、源码绑定与关键结论仍保留原处；保护不是科学接受标签。
- [Campaign 固定阅读版](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/campaigns/force_memory_closure_20261030.md)和[当前 Campaign 卡](../../../campaigns/force_memory_closure_20261030.md)保留完整问题谱系；[原追溯索引](../../../campaigns/retrospective_force_memory_20261030.md)保留编制时点与固定提交来源。
- [正式 Run 中文总表](../../run_index.md)连接已登记的 p 通道工作；字段未记录的采用决定显示 unknown，不用提议值代替 root 决定。

所有指标沿原模型、train 范围、时域和尺度解释。风险不是 RMSE，历史 21/30 漏检不作为后来 GFTD 的同期成绩；原 val/test 的暴露和关闭历史保持不变。

另可阅读[2026-10-29项目概览](project_overview_20261029_reading.html)：这是修复链接的历史阅读版，字节不变原件与来源快照另存。恢复评价和p通道检查的[旧代码入口](../../../experiments/history/force_memory/README.md)按原报告回查。

第八批在前序代码／概览归档后复核保护边界，再解除6份报告／证据的旧路径依赖；[最新累计位置](../../../results/repository_organization/20261109_history08/migration.json)与本主题保护表接续原记录。
