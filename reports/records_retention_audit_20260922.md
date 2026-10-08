# reports / results 保留价值审查

日期：2026-09-22。工作类型：文档与依赖审查；模型晋级：无。
审查阶段仅新增此记录，未运行实验；用户批准后的整理结果见文末。

## 范围与结论

盘点两个目录的全部文件，重点阅读 13 份 E 系列报告、后续纠错与论文状态记录，
检查代码/测试/论文对旧结果的引用，并对模型文件做字节哈希比较。
没有读取 private_data、原始 TEST 轨迹或加载模型运行；未逐项重算历史实验。
因此，下述“错误”指后续已接受审计明确纠正的主张，不是本次新作科学否定。

审查前 reports 为 174 个文件、3,631,541 字节；results 为 156 个文件、5,772,247 字节，
全部已被 Git 跟踪。合计约 9.40 MB（十进制），13 份 E 报告仅 55,774 字节。
主要成本是过期结论混入当前上下文，不是磁盘空间。失败记录不等于无意义记录。

## 已确认需要退出当前结论的表述

| 记录 | 问题 | 后续有效解释与依据 |
|---|---|---|
| [E2 剪枝](E2_modular_prune.md)、[E6](E6_fold_confirmation.md) | 旧剪枝在真实物理时域和完整模块采用门修复前产生；“五折全保留三族”不能作为当前结论 | [0001](accepted/0001_pruning_horizons_and_module_adoption.md) 修正端点/采用门；[0009](accepted/0009_five_fold_refit_acceptance.md) 中仅 F1/F3/F5 采用，F2/F4 回退 |
| [E4](E4_certificates_margin.md)、E6、[E8](E8_margin_frontier.md) | “综合证书 PASS”“获证书判稳”等过度表述 | [0002](accepted/0002_candidate_energy_not_certificate.md) 明确局部优化候选不提供全局逃逸能量下界；只支持局部结构检查和启发式筛查 |
| [E5](E5_oscillation_frequency.md)、[E5b](E5b_frequency_crossfold.md)、[E5c](E5c_damping_crossfold.md) | 旧 7.2% / 0.04 标题不能作为修正模态协议的结论；涉及选择期验证、旧模型与有利匹配统计 | [0010](accepted/0010_project_conclusion_audit.md) 报告修正后成功事件频率误差 19.49%、阻尼误差 0.07537、MAC 0.37647，且有 10/177 失败；不是同一口径直接排名 |
| E6 标题与第 1/3 节 | “一次性 OOS”“test 只读一次”与后续实际访问记录不符 | [0003](accepted/0003_holdout_lifecycle_and_modal_reporting.md) 记录 E6b/E6c/E8 重开同一 build；旧 315 事件仅为已消费的探索性证据 |
| E4/E6/E8 的 0/51 | 不能推为零风险或安全保证 | 0002/0010：仅 9 个 source/fault-line 组，既有审计给出单侧 95% 上界约 0.283 |
| [paper_release_candidate_v1](paper_release_candidate_v1.md) | 顶部仍称“英文稿可投稿就绪”，并保留一次读取的判断 | [paper/STATUS](../paper/STATUS.md) 明确旧正文/图/PDF 未纳入后续重拟合和模态结果，不可作为当前数值报告 |
| [accepted/0008](accepted/0008_f1_prune_rerun_acceptance.md) 第 42–46 行 | 从最终系数字节相同推断尺度错误从未影响中间删除路径，论断过强 | [0009](accepted/0009_five_fold_refit_acceptance.md) 第 44–49 行已明确纠正；accepted 目录不代表其中每句话永久有效 |

E0/E1/E2-backbone/full 属于历史数据检查与开发基线；E3 的 Pm-ODE 和 E7 的基线比较
也保留为当时条件下的开发诊断。没有依据把这些文件整体判成“造假”或“无价值”。
旧错误的发生、发现和修正链本身需要保留，不能删除错误记录后只留下成功叙述。

## results 分组处置建议

| 目录组 | 文件数 | 字节数 | 建议 |
|---|---:|---:|---|
| e00_data_check、e01_m0_f1 | 4 | 13,419 | 历史数据 QA / 基线，退出当前成果导航，保留 |
| e02_backbone、e02_full、e02_prune | 10 | 161,189 | 修复前历史模型；有现存测试/绘图依赖，保留原路径 |
| e03、e03b | 20 | 1,065,332 | 历史 Pm 诊断，有少量相同模型字节；保留完整阶段记录 |
| e04、e05、e05b、e05c | 4 | 158,223 | 证书/模态旧语义已降级，仅供历史追溯 |
| e06 | 51 | 898,405 | 修复前折模型和已消费 test 聚合；测试、绘图依赖，保留且禁止重新解释为确认性结果 |
| e07、e08、e09_valdifficulty | 11 | 120,720 | 旧比较、角度敏感性和难度描述，历史开发/探索证据 |
| e10_refit_v2 | 50 | 1,913,132 | 修正五折重拟合记录，保留；F1 旧中间阶段标准化摘要限制见 0009 |
| e11_modal_validation_v2、e12_refit_audit | 2 | 565,392 | 修正模态与重算汇总，保留；不是全项目当前模型或最终确认 |
| e13_failure_diagnosis、e14_force_balance、e15_electrical_balance | 3 | 873,317 | 有价值的失败机制诊断，保留；离线 Pe 替换不等于部署预测 |

另有 results/README.md 3,118 字节。results 没有发现临时日志、缓存或 scratch 文件。
10 个继承 JSON 含 NaN/Infinity；[results/README](../results/README.md) 已说明来源。
不能为“清理”而改写这些快照，也不能据此把其全部数值视为错误。

## 删除或移动前必须处理的依赖

- [test_module_adoption.py](../tests/test_module_adoption.py) 第 38–68 行直接读取
  e02_backbone/e02_full 摘要、e02_backbone 模型及 e06/F4 摘要。
- [paper_figures.py](../experiments/paper_figures.py) 第 217/282 行读取 e06 的
  test_margin 聚合及 e02_prune 模型。旧图生成能力不等于当前科学认可。
- [summarize_refit_v2.py](../experiments/summarize_refit_v2.py) 第 34–42/88 行读取
  e10/e11 并核对模型与依赖哈希。run_modal_validation 和 e13–e15 诊断也使用 e10。
- [verify_artifacts.py](../experiments/verify_artifacts.py) 按目录扫描 results 下 JSON；
  更改目录结构也会改变其审查范围。没有字面路径引用不等于没有动态依赖。

41 个 NPZ 与 2 个 PT 模型的哈希比较发现 11 组重复，理论上去掉组内多余副本仅省
45,468 字节。例：e03 的 gmd_backbone / m03eg / pm_backbone_backbone 与 e03b 的
pm_long_backbone_backbone 四个 NPZ 相同；部分 e06 与 e10 的 F2–F5 模型也相同。
相同参数不等于相同实验记录；sidecar、尺度、采用门和阶段来源可能不同。
尤其 e10 的 backbone/prune 相同可能记录了回退决策，不应作为垃圾删除。
Markdown 全文哈希未发现完全相同副本；下列冗余属于语义和流程上的重复。

## 可清理候选与顺序

1. **优先修正导航与失效标记。** reports/README 仍把 accepted 目录和“最新 accepted”
   作为权威捷径，并把其余记录笼统归入历史观察，无法准确涵盖现行预注册和 rejected
   科学证据。入口应指向 current_state；E 报告与旧论文就绪报告可加“历史/已被取代”
   提示和纠错链接，保留原文。results/README 的旧 e 系列说明仍有用，但不代表当前研究阶段。
2. **可退出工作树的低价值过程快照候选：**
   [gate_status_audit_v1](history/gate_status_audit_v1.md)、
   [manuscript_rewrite_inflight_verify_v1](history/manuscript_rewrite_inflight_verify_v1.md)、
   [paper_tex_quality_v1](history/paper_tex_quality_v1.md)。
   它们分别记录已完成的早期门控建设、改写中途状态和旧版排版修复。
   先保留可恢复 Git 提交与有效结论链接，再处理 incoming references；
   0004 仍引用 inflight，paper_release_candidate 仍引用 tex_quality，故不是立即安全删除清单。
3. **历史归档候选：** 13 份 E 报告，以及 paper_draft/figs_bib/number_audit/
   release_candidate_v1、synthesis_paper_main 等旧论文装配记录。
   从日常恢复和成果引用中移出即可；移动文件需同步内部相对路径与引用。
4. **保留：** accepted/failed 的有效决策与纠错报告、预注册、e10–e15 完整证据链，
   以及有现存消费方的 e02/e06 产物。0078 的实现失败记录和固定预算失败也不因失败而删除。

没有发现值得立即破坏证据链去删除的大块垃圾。建议先治理“哪些结论还能引用”，
再考虑历史文档是否退出工作树；不值得为约 45 KB 模型副本引入去重系统。

## 用户批准后的整理结果（2026-09-22）

- 重写 reports/README 导航，指向 current_state、有效预注册及完整接受/失败证据；
  results/README 明确 e 系列是历史产物，不是当前模型目录。
- 给 13 份 E 报告、5 份旧论文装配记录、3 份旧改写/治理分析加历史提示与纠错链接；
  accepted/0008 增加局部推断已被 0009 纠正的说明，原验收结论保留。
- 三份过程快照移到 reports/history，并加历史提示；旧版论文交接索引、0004 和
  本审查的引用已同步。快照原文完整保留，未以删除历史来隐藏错误。
- 数值 JSON/NPZ/PT、模型路径、预注册阈值与研究代码均未因本次整理改变。
  没有去重模型，没有运行实验或读取新的 TEST 轨迹。
- 验证通过：29 份整理文档中的 101 个本地链接全部有效，无尾随空格，
  `git diff --check` 通过；三份归档快照除新增提示外原文一致，
  results 的变动仅限 README，所有数值文件保持原状。
