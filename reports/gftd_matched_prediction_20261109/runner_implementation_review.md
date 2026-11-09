# 匹配预测 runner 实施审查 v1

结论：当前版本暂不接受执行。发现三项需要在资格之前修正的问题：资格计量可能在 profile 异常后失效；拟合异常没有遵守新增的成本未知标记；评分器的间接源码依赖未完整加入冻结清单。其余已核对的阶段顺序、原分母、正式初态与主细预测、参考比较和 L2/L3 调用范围符合冻结设计。修复限于计量、异常回执和源码身份，不改变科学规则或增加科学计算。

本次完整读取 runner、资格驱动、测试、接入说明和冻结协议，并读取评分器及相关公共调用点。没有导入被审模块、执行测试、模型、ODE、证书或读取原始数据。共享适配模块由另一位 Reviewer 审查；其候选门与罚项修复不在本报告重复裁决。

## 冻结对象

| 文件 | SHA-256 |
|---|---|
| `experiments/gftd_matched_prediction_20261109/run.py` | `d5a354d50a7382ac4d1753afb37f6a97876d23c9fe7426def922d6c45138b890` |
| `experiments/gftd_matched_prediction_20261109/qualify.py` | `287d46730699cdf46b6318aa793e6ee16040830ef5b27b29bb10545f398062e2` |
| `tests/test_gftd_matched_runner.py` | `1b30ee029680382ccfea8144dff58860959cfdf17802fcd820b6bfeaf513d0c2` |
| `runner_integration_notes.md` | `3d4e5e9a3281a6a2231d442dcae010079a74734c4f527301e78ab463a8cee5b3` |

实际哈希与冻结值一致。本审查绑定协议 `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43`、设计审查 `8322c6791a192b9bcd0e881d7b53078527d3f17ccad61f1b14fdcd8562972b5b`、算术澄清 `c73f38d66c56fbaef2cf8bc4fce9ea7a978fe72ab56cf778778ee291842582d8`，以及已接受的隔离分析实施审查 `9c360b0b7fdda819765fb70512b0441e426d39184bcff682ec5da59bd8039cc3`。

## 执行阻断与最小修正

**R1：资格计量器会因 profile 抛异常而失效。** `qualify.py:97–104` 在发现禁止函数代码时，从 profile 回调抛出 `RuntimeError`。Python 会撤销抛异常的 profile；pytest 可以捕获异常并继续后续测试，而 RHS/H 的 guard 又依赖 profile 更新的 `counts`。这样在最终 `meter_lost` 检查前，后续调用可能不再被正确计量，不能保证共享 128 RHS／2000 H 硬上限。

最小修正：RHS/H 使用独立、持久且调用前递增的 guard 账本，禁止的实际积分／优化／数据入口由方法和 alias guard 阻断；profile 只观察和记录，不抛异常。一旦发现未知禁止 alias 或 profile 丢失，所有后续实际 RHS/H 都须拒绝，不能让 pytest 捕获后继续模型工作。保留跨 attempt 额度、先 claim 后执行、未知成本不自动复查的现有规则。用 mock 检查计量失效与阻断路径，不增加物理点或人工轨迹。

**R2：拟合异常未检查 `cost_unknown`。** `run.py:617–629` 将某些带数值异常原因的 `FitFailure` 转为正式 `fit_failed`，但未读取 `diagnostics.cost_unknown`。共享模块若有尚未归入 aggregate 的局部调用，再以数值异常退出，当前 runner 可能把不完整计数当作完整数值拒绝，继续后续阶段。

最小修正：只有明确 `cost_unknown=false`、计数已对账且符合登记的普通数值失败，才能完成为 `fit_failed`。未知成本或缺失此保证时，先保留已知账本和诊断，标记 partial/unknown 并停止受影响阶段；总实际成本保持 null，不补跑、不借余量。阶段对账、冻结和下游读取也须拒绝这种未闭合状态。正常成功路径同样需要明确其成本状态。L3 若未返回内部计数即异常离开，也应保留成本未决；现有 interrupted 状态不能被解释为已完成的零 L3 调用。

**R3：评分源码闭包遗漏间接依赖。** `run.py:77–83` 的 `SCORE_SOURCES` 包含 `train_joint_latent.py` 和 `run_private_m2.py`，但缺少二者直接导入的 `src/kiki/data/private_dataset.py`。正常 prepare 尚未导入评分器，该文件不在 `imported_sources()`；资格中的 `test_frozen_l1_units_los_full_grid_and_denominators` 导入评分器后，会在最终 `verify_sources()` 被识别为未冻结依赖。评分阶段也存在同一问题。

最小修正：将该文件加入显式评分闭包，并用纯静态导入关系检查完整链。无需为补齐源码身份导入旧数据入口、打开数据或消耗一次实际资格。现有评分函数的调用范围保持不变。

以上问题修正后需要新的冻结哈希及追加审查。本 v1 结论和源字节应保留，不覆盖成已通过记录。

## 其余边界核对

准备阶段只读取公共元数据、源码及归档字节，不打开父 checkpoint、合法观测或未来。它绑定前轮完整执行与 26 文件归档，并核对归档路径及 prepare 哈希；前轮归档格式与 runner 所要求字段一致。每阶段有独占 started claim，逐 Job 有独占 claim，阶段异常后不自动重放。

24 个全局索引由固定 `floor(j*297/24)` 生成，local_slot 按固定顺序编号并绑定原 event_id。全297原索引、唯一 ID、267/30 标签计数及顺序哈希先验证；实际进入 fit 的包只有五个合法字段。两父的非边物理量、中心和冻结尺度先比较，拟合后再次核对核身份、模式、固定量、配置及诊断哈希。

拟合最多两个独立 worker；部署可按 root 登记使用八个 worker。spawn 初始化逐进程核对源码和环境，每个 worker 内同步执行单个初始化，两个最终模型分别按 mode 保存，不会把两模型参数写入同一对象。调度只提交固定 Job 列表，没有重试路径；异常后取消未开始任务，已启动工作由收尾对账保留。线程数固定为一，worker 与宿主时钟分别记录。

部署调用一次公开 initialize，包装 `_integrate` 直接捕获原六门的六腿返回数组，以同一 functional、DP8 方法、端点字节和门套数绑定，并在 finally 恢复包装。历史 residual 调用与六门调用分别核对 600/1200 及每包最多两套上限。正式点必须与所选 best、theta/omega/p₀ 硬锚及原六门一致，stage2 失格不在 runner 内回退。参考和长时预测都从这个已保存正式点开始，不重新初始化。

Radau 回调上限 24000 覆盖构造、数值 Jacobian、step 和 dense 操作；主、细比较各自使用连续有限共同网格，全段未完成时确认保持 false。额外严格门没有回写六门资格或拦截长时轨迹。15 秒 DP8 主／细各只运行一次，保存 1501×50／3001×50 全部内部步及失败后的未获得区域，实际 RHS 在回调前计数；序列化前保存已完成计算账本。

两模型冻结完成后才部署；全部部署选择和逐 Job 身份冻结后才允许 score。score 显式绑定原未来缓存哈希，仅读取 metadata 与对应事件状态，核对原顺序、标签及与合法100 ms历史的对应。评分调用旧纯 NumPy 入口，不调用其训练、rollout 或数据生成入口。细轨迹抽取原网格评分，主轨迹同范围单列；初态字节一致性、全50维有限性、LOS-aware 门和实际真值时长规则由原合同保持。fit_failed 及初始化失败保留全部原 ID，不能转成 unavailable 删除；任一 eligible 失败使对应 official 为 null。

L2/L3 对每个可用最终模型各调用一次：每模型原13半径、每个正式点一次包络，共享 `TaylorBudget(160380)`；L3 保留每模型 217 RHS／980 独立梯度／2 eig 上限和实际平衡。它们绑定最终 checkpoint 与已选端点，既不使用真实未来，也不修改共享参数或选择。拟合失败模型保持 not_run，没有父模型补位。

当前测试源码使用固定数组和 mock 检查公开分派、六腿原数组捕获、claim 防重放、未来冻结门、预测失败范围、Radau 计数及原 L1 单位／LOS／分母。未执行这些测试。修复后还须实际完成冻结的资格，才能获得运行决定；本审查不释放任何科学阶段。
