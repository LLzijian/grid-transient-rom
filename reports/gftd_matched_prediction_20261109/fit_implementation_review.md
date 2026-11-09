# MRun 共享适配实现审查 v1

当前版本保留了父模型、原历史尺度和两起点选点规则，也将共享适配与部署初始化的预算分开。进入接线资格前须修正两处会改变登记执行的行为。第一处省略了数值失格候选的剩余检查，第二处用去偏后非零系数替代提案筛选掩码计罚。两项都能用固定 mock 验证。

审查范围为共享适配源码、固定 mock、公共接口的原有调用链及接线说明。核对了原字节 SHA-256。未导入或执行源码、测试、优化器、ODE 或模型，未读取原始观测或真实未来。

| 来源 | SHA-256 |
|---|---|
| [共享适配模块](../../src/kiki/models/gftd_matched_prediction.py) | `59ff581282984e1250a2c2aceeeaabf332e2e7b47a0cf7db2bd02adbc3ac1c11` |
| [固定 mock](../../tests/test_gftd_matched_fit.py) | `3c0401d950b9da9ebfed3dbe0bcbb2cf277c6c7820428489e0ad1e31cab054ad` |
| [接线说明](fit_integration_notes.md) | `8a719dd5c1ee7e6fea91bfa2317e1024de332c12cbcb19b736082699b95c9ee5` |
| [有限协议](../../campaigns/gftd_matched_prediction_20261109/protocol.md) | `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43` |
| [设计独立审查](design_review.md) | `8322c6791a192b9bcd0e881d7b53078527d3f17ccad61f1b14fdcd8562972b5b` |
| [父设计草案](../gftd_bounded_initialization_20261109/next_mainline_design_draft.md) | `530b9e34bf042c81acee58d5410c760e0457c571c23d8f1d2afd6106d90029cd` |
| [人工资格澄清](qualification_clarification.md) | `c73f38d66c56fbaef2cf8bc4fce9ea7a978fe72ab56cf778778ee291842582d8` |

## 接线资格前修正

1. **数值失格候选仍须完成其余已选端点的六门。** 24 包均有合格有限选点而第 2 包数值失格时，当前循环在第 2 套后返回，后 22 套保留 `not_run`。固定 mock 把这项提前停止写成预期。协议 §3 的逐包检查适用于已有 24 个有限选点的候选，root 在本次审查确认该执行口径。应逐包保存原选点的六门及计数，汇总失格后再决定候选状态。首候选失格仍终止对应模型的拟合，后续提案失格保留 incumbent。缺任一有限选点的候选仍不启动六门，任何失格均不换起点。将现有失格 mock 的预期改为完整 24 套且选点顺序不变。 [协议 §3](../../campaigns/gftd_matched_prediction_20261109/protocol.md#3-两模型的有限共享适配) [模块第 137 行](../../src/kiki/models/gftd_matched_prediction.py#L137) [测试第 79 行](../../tests/test_gftd_matched_fit.py#L79)

2. **提案按筛选掩码计罚。** 当前目标在父候选和提案上都用去偏后的 `c > 0`、`beta > 0`。旧去偏可在仍开放的列返回精确 0，并标记阈值不一致。该提案仍可进入历史 MSE 比较。若父候选和提案的 MSE 相同且各登记开放一列，去偏将提案列置 0 时，现实现会给提案少收 `1e-12`，超过 `1e-14` 的严格改善余量。按父设计及 root 确认，首父按有效非零掩码，提案按 `backward_select` 返回的筛选掩码，包括去偏后系数为 0 的已开放列。候选各保存边和调制掩码。固定 mock 用同 MSE、同开放数和一个去偏零列确认保留 incumbent。 [父草案第 32 行](../gftd_bounded_initialization_20261109/next_mainline_design_draft.md#L32) [模块第 130 行](../../src/kiki/models/gftd_matched_prediction.py#L130) [旧去偏第 460 行](../../src/kiki/models/gftd_fitting.py#L460)

## 已核对的行为

- 首候选取父 functional 的独立副本。原指纹与内层 fit297 配置相互验证，首评估前不重写分数门。每个提案用同一父的非边常数建模，重新开放静态 45 列或有界 90 列。历史字典沿用独立于门的 `(A_static,A_K)`，以 `(c-beta,beta)` 表示。必要 NNLS 比较失败由旧选择器结束提案，未决提案和不改善提案均保留先前的模型、配置和选点。 [模块第 290 行](../../src/kiki/models/gftd_matched_prediction.py#L290) [选择器第 475 行](../../src/kiki/models/gftd_fitting.py#L475)
- 请求固定 297、K10 和 24 个全局索引。只在取包时用全局索引。准备、代理、字典与审计按本地槽顺序操作。两核准备缓存都是单次 fit 的局部对象，不触及旧进程全局缓存。新准备键绑定请求、24 包、网格与 DP8 系数。 [模块第 49 行](../../src/kiki/models/gftd_matched_prediction.py#L49) [模块第 300 行](../../src/kiki/models/gftd_matched_prediction.py#L300)
- 两起点为零和原固定交替向量，每起点一次最多 1200 个实际残差。只在 `converged` 或 `budget_limited` 且选点及历史有限的结果中比较 MSE。同值保留先到起点。保有有限 best 的 `failed` 按设计审查的明确条件排除。各候选先完成 24 包选点，缺包时目标无效且六门为零。 [模块第 99 行](../../src/kiki/models/gftd_matched_prediction.py#L99) [设计审查第 18 行](design_review.md#L18)
- 同点诊断复用保留的原尺度残差，通过相同的端点映射与主步 DP8 求两档差分。投影符号、边界容差、范数分母、方向试算和无变化跳过均与协议一致。诊断没有回填选点。个别步长失格由该行及 `locally_consistent=null` 表示，汇总时应保留点诊断未决。细历史条件信息使用最终选点，列数是完整 45/90。信息不足时支持保持未决，L1 可沿合格方程继续。 [模块第 159 行](../../src/kiki/models/gftd_matched_prediction.py#L159) [模块第 198 行](../../src/kiki/models/gftd_matched_prediction.py#L198)
- 正常数值失败在旧优化器或六门内部返回已用计数。输入参数和候选副本独立，公共 fit 对输出配置、JSON 可序列化性与状态验收后才更新模型。若技术异常穿出内部计数器，外层已有计数是已回收下界，runner 须将无法核对的调用标为未知，保存错误及已回收计数并停止受影响阶段。当前隔离模块的 `execution_unresolved` 不能当作普通候选数值拒绝。 [模块第 383 行](../../src/kiki/models/gftd_matched_prediction.py#L383) [公共 fit 第 394 行](../../src/kiki/models/gftd_kuramoto.py#L394)

## 预算与接线验收

两模型的结构上界与协议一致，前两项修正不增加已登记机会。代理为 `2×4×24×2=384` 个优化任务、460800 个实际残差、221184000 RHS。六门为 `2×4×24=192` 套，每套 6480 RHS。最终诊断按每个 10 维点最多 44 次、20 维点最多 84 次计，合计 3072 残差和 1474560 RHS。细历史再积 46080 RHS。准备为 96 个记忆 SVD 和 46080 次端口回调，单列在 ROM RHS 之外。信息审计为 100 次基础谱分解及 270 次条件投影对应的谱分解，共 370 次信息 SVD。每个完整库提案至多 8191 个代数项，6 次合计 49146，实际 NNLS 次数不超过代数项数。 [协议 §9](../../campaigns/gftd_matched_prediction_20261109/protocol.md#9-有限调用上限异常和闭合)

1. 旧源归档后增加互斥 RunContext 分支，公共 fit 仍先验证五字段、时间和 sigma，继续走原输出提交路径。只对 MRun 使用 1200×2 的共享预算，fixed-DP8 内层初始化合同继续控制 600/1200。注册 runner 须验证父 checkpoint、输入、事件顺序和源闭包的实际字节。模块的请求字符串与 functional 指纹校验只验证传入对象与声明的一致性，文件来源由 runner 绑定。 [接线说明](fit_integration_notes.md) [公共 fit 第 370 行](../../src/kiki/models/gftd_kuramoto.py#L370)
2. `on_port_rhs`、`h`、`capture` 均以仅限关键字的可选参数接入。无回调时保留旧四参数端口调用，计数回调在端口算术前执行。`h` 默认保持旧差分值。六门捕获在原积分返回后、观测转换前登记，保持原步长、次数和门槛。阶段 D 应从冻结端点生成参考与正式主细轨迹，公共 `rollout_state` 会再次调用 `initialize`，不能用来重算已冻结初态。 [端口第 107 行](../../src/kiki/models/gftd_discovery.py#L107) [六门第 290 行](../../src/kiki/models/gftd_fitting.py#L290) [公共 rollout 第 516 行](../../src/kiki/models/gftd_kuramoto.py#L516)
3. 接线后补齐协议已登记的 mock 资格。现有候选测试整体替换了 `_prepare`，能验证取包索引，双核两网格实际缓存隔离尚未覆盖。公共 fit 的互斥、白名单、预算和方法分派，以及旧 exp / bounded v3 存取身份，也要在接线源上核查。固定测试可 mock 端口、优化与积分。诊断多项式、数值失败和拒绝后保留旧候选沿现有输入验证。半径澄清的 `7/15` 只约束 L2 固定算例，不涉及共享拟合逻辑。 [固定 mock 第 104 行](../../tests/test_gftd_matched_fit.py#L104) [协议 §5](../../campaigns/gftd_matched_prediction_20261109/protocol.md#5-人工资格固定输入无ode或真实优化) [资格澄清](qualification_clarification.md)

决定为先修正第 1、2 项，再对接线后的冻结闭包执行登记资格。该决定只针对代码。模型效应和科学判断等待实际阶段证据。
