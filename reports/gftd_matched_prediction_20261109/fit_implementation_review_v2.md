# MRun 共享适配实现复审 v2

v1 两项缺陷已修正，公共 fit 接线符合冻结协议，接线后的共享模块可进入固定资格。正式释放仍需 runner 正确处理未知成本，并完成既定的存取测试和资格计量。

本次按冻结哈希对照 v1 原字节、v2 源与 mock，以及四处公共接线。所有判断来自静态阅读，测试结果尚无。本次未运行 import、测试、ODE 或模型，未读取原始观测或真实未来。

| 冻结来源 | SHA-256 |
|---|---|
| [v1 独立审查](fit_implementation_review.md) | `a0f654585e7b79df682b69df0a48db02d8894450f740a6a93212d0d800f7a8d5` |
| [BRun 原源码归档清单](../../results/gftd_bounded_initialization_20261109/source_archive.json) | `b180c72e5748fa472621e17029283db4de264fe85c9ad2b8a0dfcbfe4a91c026` |
| [共享适配 v2](../../src/kiki/models/gftd_matched_prediction.py) | `5cfc622a1d7d8bea5ed195b115417ec60ce0d4bbcd285ea7c2a0b81022d6b5cc` |
| [共享适配固定 mock](../../tests/test_gftd_matched_fit.py) | `6308ba857b4c2ddba4d45bfc2b73631d8d91e3ce563326bcf9dea05eeeb8410d` |
| [接线说明 v2](fit_integration_notes.md) | `c9467826f03e1310af191c2d366b49f0d54c0ff43d36626d7079dc0c06a2dbe4` |
| [RunContext](../../src/kiki/models/base.py) | `026a5e42e3b074f5a535bdf1fd4d7458ac6e05aad1f6c567e77429cee60631d1` |
| [公共 GFTD fit](../../src/kiki/models/gftd_kuramoto.py) | `c503b54e609681b3b94aed0c31c47839c17851efdf2a8070cd93ccd34ab8d520` |
| [端口回调](../../src/kiki/models/gftd_discovery.py) | `fe6035eb92aeeb383c2b97ce50b39123261ac9dd2ee8f50379cb9f79064de5b3` |
| [差分与六门捕获](../../src/kiki/models/gftd_fitting.py) | `423428b3427eaa00fc10bf2467b04f6751a6e922f66cf0d43c16b5ba4de8da26` |
| [runner 固定测试快照](../../tests/test_gftd_matched_runner.py) | `1b30ee029680382ccfea8144dff58860959cfdf17802fcd820b6bfeaf513d0c2` |
| [存取及 DP8 固定测试快照](../../tests/test_gftd_bounded_kernel.py) | `bf62fc758af178a6490165a50e7d0ffbb1af0dc9a91e880103c94c924aef659c` |

协议 SHA-256 仍为 `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43`。v1 的模块、mock 和接线说明在 `implementation_v1/` 的归档字节与 v1 审查逐项一致。BRun 归档清单所列 26 份快照逐项复核字节数及 SHA，全部一致。归档提交 `95afe8cd187e7947c65dd30e1fb6747341d8735e` 在当前提交历史中。[有限协议](../../campaigns/gftd_matched_prediction_20261109/protocol.md) [v1 归档](implementation_v1/) [BRun 归档清单](../../results/gftd_bounded_initialization_20261109/source_archive.json)

## v1 缺陷闭合

1. 24 个有限选点齐备时，循环逐点保存六门结果、捕获数组和实际计数，退出循环后汇总候选资格。第 2 点失格的 mock 断言仍执行 24 套，并核对后续点的通过记录。缺一个有限选点时六门为零。首父失格不提案，提案失格不替换 incumbent。192 套上限仍由两模型各四候选、每候选 24 套限定。[模块第 127 行](../../src/kiki/models/gftd_matched_prediction.py#L127) [固定 mock 第 79 行](../../tests/test_gftd_matched_fit.py#L79)
2. 首父传入有效 `c>0`、`beta>0`，提案传入选择器返回的阈值筛选掩码。每候选复制两掩码，罚项用相同口径。零去偏列 mock 核对其仍在掩码和罚项中，且会改变旧错误口径产生的接受判断。完整库提案入口仍开放全部 45/90 列，父非边参数、原历史尺度、两起点和严格比较公式沿用。[模块第 105 行](../../src/kiki/models/gftd_matched_prediction.py#L105) [模块第 310 行](../../src/kiki/models/gftd_matched_prediction.py#L310) [固定 mock 第 89 行](../../tests/test_gftd_matched_fit.py#L89)

新增双缓存 mock 直接调用 `_prepare`，替换端口与 SVD 本体，核对两核、两网格、24 槽和 48 次准备及记忆分解、23040 次端口回调的单模型计数。技术异常 mock 核对 `execution_unresolved`、`cost_unknown=true`、已回收计数及停止提案。各上限和原尺度同点诊断按 v1 已核对的公式保留。[固定 mock 第 105 行](../../tests/test_gftd_matched_fit.py#L105) [固定 mock 第 181 行](../../tests/test_gftd_matched_fit.py#L181)

## 公共接线与剩余资格

- 新 RunContext 字段位于末尾并默认空。公共 fit 先执行原五字段、时间、sigma 与 val 检查，再判断互斥分支，传入参数副本并沿原输出提交路径。旧两分支仍走原方法。端口回调在完整向量 RHS 算术前触发，缺省调用保留四个原参数。Jacobian 的可选 `h` 默认仍为 `2.5e-5`。六门捕获记录原调用返回数组的副本，步长、积分次数、转换、数值门及缺省返回合同保持原样。[公共 fit 第 370 行](../../src/kiki/models/gftd_kuramoto.py#L370) [端口第 107 行](../../src/kiki/models/gftd_discovery.py#L107) [差分第 189 行](../../src/kiki/models/gftd_fitting.py#L189) [六门第 290 行](../../src/kiki/models/gftd_fitting.py#L290)
- runner 的固定测试验证公共 fit 互斥、白名单、val 拒绝、参数副本及原六腿捕获。登记的 bounded-kernel 测试验证旧 exp 载荷、bounded v3 降级拒绝、DP8 配置存取、600/1200 初始化与积分分派，以及固定 ii/C 点的存取、深拷贝与规范身份。公共 fit mock 目前返回占位 `config` 与 `fit_state`，未联合验证新 fit_state 和真实可校验 DP8 配置的保存前后哈希。协议 §5④ 的该项可在同一 mock 返回合法配置并存取后断言，核对共享策略 1200 和内层兼容字段 200 分开保存。这是资格覆盖项，涉及零模型调用。[公共 fit mock](../../tests/test_gftd_matched_runner.py#L11) [bounded v3 与分派测试](../../tests/test_gftd_bounded_kernel.py#L72) [固定点身份测试](../../tests/test_gftd_bounded_kernel.py#L224)
- runner 快照 `d5a354d50a7382ac4d1753afb37f6a97876d23c9fe7426def922d6c45138b890` 的异常分支尚未消费 `cost_unknown`。例如技术异常以 `FloatingPointError` 穿出候选或诊断时，v2 会标 `execution_unresolved` 和未知成本，runner 却可根据 cause 接受为终态 `fit_failed`。修复时先检查 `cost_unknown`，保留已回收成本为下界并保持该 Job 的完整成本未知，停止受影响阶段。普通 `fit_failed` 或已知成本的登记数值失败仍可结案。该项与 runner 资格计量一并收口。[模块第 383 行](../../src/kiki/models/gftd_matched_prediction.py#L383) [runner 第 617 行](../../experiments/gftd_matched_prediction_20261109/run.py#L617)

复审决定为接受共享适配 v2 与四处公共接线。联合存取覆盖和未知成本处理由 runner v2 及其独立复查落实。资格与正式阶段释放由其实际冻结闭包和回执判断。
