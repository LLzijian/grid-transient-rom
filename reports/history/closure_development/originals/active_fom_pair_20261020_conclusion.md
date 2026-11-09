# 新条件完整档：采集环境失败，保留工程证据

本轮完成了可复现的约束条件设计与专用入口合成检查；真实短历史采集发现GPU环境缺少ANDES，48个条件、96个步长调用均在FOM启动前返回`ModuleNotFoundError: No module named 'andes'`。合格历史为0；没有执行D、完整train采集、追加更新或新val。本档关闭为环境接线失败，原结果不重试或改记成功。未采用模型、未合并本轮代码，默认模型和main保持不变。

依据[实际Astra6原方案](astra6_active_fom_20261020.md)、[执行前修订](astra6_active_fom_20261020_design_amendment.md)、[root协议](active_fom_pair_20261020_protocol.md)及[独立实现挑战](active_fom_pair_20261020_implementation_review.md)。实际源提交7c2dc8cb0a80a1367f09e5686d9e71e826a81d1f；目录private_data/closure_mechanism_20261019/P3/active_fom_pair01。原test关闭，TASK4暂停，真实生成结构搜索关闭。

## 成立的工程证据

旧条件排重后456个组合，整数动态规划冻结24条val设计及48候选，层配额分别8/8/8和16/16/16，随机12在真实采集前锁定。设计文件SHA256 c475a3bf11aaa17e2a3ca4642291a43d863541be14ec0fefdec5cd072de8c126。[条件与精确概率记录](../results/active_fom_pair_20261020/design.json)不包含新val成绩。

唯一合成QA通过9项采集检查、4组fit检查，含3次微型GPU更新的Adam与CPU/CUDA随机状态恢复核对；不代表完整256更新恢复。QA真实外层13.956230秒、条件准备7.6523984秒，均内外exit0、来源未变、清理核实。[QA证据](../results/active_fom_pair_20261020/qa.json)。生产缺少ANDES说明该合成检查不覆盖FOM运行环境可用性，不能作为真实采集验收。

## 真实采集与费用

短历史外层40.6145747秒、内外exit0；该退出码表示48条尝试及失败记录完成，不表示物理仿真合格。实际worker42944、FILETIME134369126916158413已退出，自有树清理核实；211个明确绑定文件前后一致。每次失败都保留，没有把环境失败记作物理失稳，也没有补位、补尾或缩小48分母。

全部实际原生费用62.2232031秒，加预登记3250保守支持，结项3312.2232031秒；取消本档未用36687.7767969秒回原P3。父累计57325.6209421秒，余额201874.3790579；P3余额88361.7061617。其他类别与旧900支持hold不动，子进程未重复记费。见[结果](../results/active_fom_pair_20261020/short_result.json)、[48条完整失败记录](../results/active_fom_pair_20261020/short_attempts.json)及[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

## 决定与实际下一步

本轮main集成和默认采用均撤回，代码、条件、QA与全部环境失败留在codex/active-fom-pair-20261020归档分支；无需revert，因为尚未合并。此结果没有检验两结构在新激励上的科学差异。

父授权仍有资源。root下一步实际调用gpt-6-astra，对已登记FOM环境与GPU训练环境做有界只读规划，决定独立的有限环境修复及后续合法实验。原48失败、原40000档关闭和费用不覆盖、不重置；修复不能冒称本档成功，也不能借新编号无条件重跑。继续范围和条件复用政策须在执行前由root明确登记，保持全部原科学门槛。
