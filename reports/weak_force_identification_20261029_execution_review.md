# D/W执行实现：独立有界静态审查

2026-10-29。实际模型 `gpt-6.1-sol`，独立审查者，未负责实现。仅静态读取新study/execution及两个测试定义，并核既有原生Job函数签名/回执条件；未读私有事件、实际参数或作业，未执行测试、fixture、矩阵、CUDA或optimizer。按作者停止后的四文件SHA收口；release仍UNKNOWN，真实计算继续未放行。root保留最终决定。

**结论：原976有限执行及主要科学接线成立，但正式训练准入的证据复核还需三处最小补齐。** 这些是准入证据检查缺口，不是已发生错误或科学模型失败；不要求重做数值数学或新增框架。

## 必须补齐的准入检查

**Q1：QA通过声明尚未与真实72次journal闭合核对。** `weak_force_execution_20261029.py:159–161` 仅要求eligible为D/W、两条PASS和每条宣称36；未核models恰各包含唯一D/W，也未将真实 `counts(base)` 的qa attempted/completed核为72。:179–180只拒pending。`weak_force_study_20261029.py:292–315` 的当前循环确实实现18对16+2，但真实账本应独立成为正式准入条件，不能只凭result字面字段。

最小修正：在QA阶段证据复核/进入study前，若声称配对训练准入，则核唯一D/W各一、每组journal所属两条QA轨迹真实18+18（或同等明确的每组总36），qa总attempted=completed=72、无pending，且没有formal提前调用；失败/部分QA保留实际数目但不训练。仍使用现有call_id/run_id/category/step，不增加账本系统。加入必要纯mock反例即可，不运行真实QA来验证守卫本身。

**Q2：D0/A96跨模型文件没有成为显式训练资格条件。** study:243–252正确复用旧A96fine缓存与D0新fine缓存，核各一次297身份/历史并调用原双向物理评分，输出 `D_A96_compatibility.json`，且当前局部失败会阻止该D的admission。因此没有发现源预测被额外重算。问题是execution:154–158只复核CPU/CUDA/backward/fullfit几个summary布尔量，没有显式复核cross文件和其独立qualified事实；它可以把不含cross资格的summary当完整准入证据。

最小修正：qualify summary中为D写cross资格flag及实际文件path/SHA；controller复核路径/SHA、qualified、原schema别名 `R_N=frozen_A96,A=current_D0`、source旧evidence SHA、297唯一注册ID及8s/actual15双向原门和全细子步有限性/范围，纳入后续phase.inputs；缺失、false、范围/身份错误拒绝训练。现有:170–178递归哈希会把实际文件绑定到后续Job，值得保留，但哈希存在不能替代其资格语义检查。R_N仅复用既有profile，不额外预测来源。

**Q3：新阶段读取的旧CPU数学资格JSON未显式纳入冻结。** study:228–230读取 `results/state_gated_memory_20261028/model_pre_freeze_qa.json` 并据shared/A映射准入；execution:29–34的SOURCE_ROOTS/DOCUMENTS未列它。Python源码闭包不自动包含这一JSON，parent_proof仅复用旧receipt/binding关系和runtime文件，不能等同于将当前这个读取对象的字节绑定到新Job。

最小修正：在新binding中按已核旧CPU资格身份显式纳入该JSON及SHA，并保持其CPU-only、training_admission=false、旧失败历史；不重新跑旧模型资格，也不修改旧证据。真实R_N profile来自已固定父qualification summary，A96 profile及旧fine证据也已有新freeze哈希路径，不因本项重评它们。

## 已核关键接线与唯一入口澄清

- study:18–28、45–78限定29311/29312、D/W、每正式run≤192、总976及qa72/formal768/repair72/replay64；正常journal仅qa/formal，reserved不自动启用，pending不能继续optimizer。attempt在step前，completed在真实step后，失败不补次数。
- study:88–103只载A96同一A/主history.0025，D保原47参数，W才覆盖一次弱解raw；旧Adam不加载。:139–176真实主NNLS通过一次attempt文件独占消费，隔点不求解，异常写summary。W拒绝则qualify:225–226退出，qa无eligible，study:405–406没有配对就不训练D，符合root收缩后的协议。
- study:116–135冻结同种子同schedule，:354–399将variant/seed/source/schedule身份传入原checkpoint；每16保存真实模型/Adam/RNG/覆盖位置。QA使用原SGM子类恢复及精确快照/RNG与非零Adam矩检查；没有将W变回D或内部π。Q1要求的是控制器再核实际journal，不是改恢复数学。
- study:319–351保持D/W/R_N共同T、原确认末点及E/u、风险或少漏分支和全部保护；缺完整对照、未决/优化未决不跑第二种子。共同96决定先于全部train88读取；:420–445先闭合fit及效率记录才报告385/noise88，后者不反选来源。成本里程碑只用配对已有完整fit，未测不插值，也没有自动宣称等成本胜出。
- native :222–223的supervise实参与现有 `uncapped_windows_job_20261021.py:124` 签名一致，未添加process/runtime/CPU额度。沿用的receipt检查包含实际launcher/worker exit、source前后身份及清理结果；父proof核旧worker GONE并保持旧lock owner条件，不偷换PID身份。新lock也要求实际worker pid/creation time。UNKNOWN release在freeze、单phase launch和run前均硬拒。
- 曾指出的人工资格重复风险已由root在本次审查中明确解决：本轮**不单独运行core fixture CLI**，仅正式native loop的weak阶段调用一次默认固定套件（正常最多4人工NNLS）；此前可单独放行的范围收缩为纯unit。study:145因此不再是第二套运行。独立CLI继续存在不构成本轮额外授权；失败即结束，没有自动修复或重新正常起一套。本报告不要求添加复用缓存系统。

测试仅阅读定义，未运行。当前两个纯测试文件覆盖release UNKNOWN、有限循环、科学门和总账，尚未覆盖Q1/Q2所述证据反例；作者补齐这些具体守卫后可做小范围纯检查，不需重复旧数值数学套件。此审查不是持续站岗或每run仪式。

## 作者停止后的源身份

| 文件 | 实际SHA256 |
|---|---|
| experiments/weak_force_study_20261029.py | `a95bc0282690451f53d43bc0ddd9dbdbe99c0038d0fca1244849fd400dfea6bd` |
| experiments/weak_force_execution_20261029.py | `2dbe098ece4527f323ca27bacc056d11a171d9693434c3a823f44e6cdf828143` |
| tests/test_weak_force_study_20261029.py | `ff1be267ad1451d8fe8b0bba5d9693d73db2d2068856145e83533ff24a35c53b` |
| tests/test_weak_force_execution_20261029.py | `ae9eaf8a369046e548a9c09e7d3d29a433b75261b9a80c2e266aa6380c14a536` |

除本报告未改其他文件。待上述准入检查最小修订后，仅局部复核对应证据与最终SHA；在此之前不把当前实现标为真实事件执行PASS。
