# 匹配共享适配接线说明

共享适配模块直接评价两份已登记父模型。三个提案每次开放完整 K10 候选库，在 24 个固定包上用 DP8 重新选点。每包先从两个合格起点中选历史 MSE 最小的点，再运行一次原六门。首候选失格返回 `fit_failed`，提案失格保留此前的模型与选点。最终同点诊断和条件信息审计保留原尺度，不写回拟合结果。

## 请求与来源

`RunContext.gftd_matched_prediction_v1` 使用以下精确字段，值与有限协议绑定。

| 字段 | 值 |
|---|---|
| `format`, `run_id` | `gftd_matched_prediction_v1`, `gftd_matched_prediction_20261109` |
| `protocol_sha256`, `input_sha256`, `event_order_sha256` | 模块同名大写常量 |
| `mode`, `kernel`, `parent_model_sha256` | `static` 或 `modulated`，对应固定核与父文件 SHA，见模块 `PARENTS` |
| `training_count` | 整数 297 |
| `adaptation_slots` | 24 行，各行精确包含 `global_index`, `local_slot`, `event_id`，全局索引等于模块 `ADAPTATION_INDICES` |
| `parent_fit297_config` | 从已核对父文件的 fixed-DP8 初始化配置取出完整 `fit297_config`，包括父 functional 指纹、原尺度、`k_ref` 与旧准备键 |
| `source_sha256` | 非空字典，冻结路径到小写 SHA-256 |

runner 在公开 fit 之前核对实际父模型文件、源闭包、合法输入和事件顺序的哈希，验证两父的非边物理常数与原尺度一致，并从冻结事件顺序确定 24 个 ID。fit 模块检查固定身份、模式、模型指纹、索引和字段。公开 fit 的合法包仍只含历史、末点功率和 sigma，事件 ID 只用于槽位记录。模块不读路径或访问真实未来。

新准备键哈希包括完整请求、24 个合法包、网格与 DP8 系数身份。每次 fit 的 24 槽两网格缓存由该次调用独占，两核之间没有进程内共享缓存。跨进程防重放由阶段 claim/receipt 执行。最终部署配置保留父请求和父准备键作为初始化合同来源，新准备键与共享 1200 预算分别位于新 `fit_state.preparation_key` 和 `fit_state.strategy`。

## 当前运行闭合并归档后的接线

1. 在 `src/kiki/models/base.py` 的 `RunContext` 末尾增加可选 `gftd_matched_prediction_v1: dict[str, Any] | None = None`。在 `GFTDKuramotoModel.fit` 的现有输入白名单、非空和 val 检查之后增加互斥分支。两个请求均非空时拒绝；新请求非空时以本模型 functional、原验证后 packets、`mode=self.fit_mode` 和新请求调用新模块 `fit_shared`。其余分支和 FitResult、保存、加载、克隆与部署入口沿用现有逻辑。原配置验证器验证返回的 fixed-DP8 配置。
2. 在 `gftd_discovery.py` 的 `reconstruct_ports(t, omega, p_end, functional)` 与 `prepare_weak_history(packet, functional, step=.0025)` 末尾各加仅限关键字的 `on_port_rhs=None`。将 `reconstruct_ports` 中传给两条 `_time_rk4` 的 lambda 各改为同函数内的 `p_rhs(time, value)` 和 `r_rhs(time, value)`。在两个函数的入口先按需调用 `on_port_rhs()`，然后原样返回各自 lambda 的计算式。这样计数发生在端口 RHS 内任何乘法或插值之前，包含触发数值异常的回调。调用者 `prepare_weak_history` 在有回调时透传该关键字，默认仍按原四参数调用。保留 `_time_rk4` 与插值函数本身。每核最多 23040 次端口回调，两核合计 46080。
3. 在 `gftd_fitting.py` 的 `_jacobian(residual, u, low, high)` 增加仅限关键字的 `h=2.5e-5`，把函数体开头的 `h, center, columns = 2.5e-5, None, []` 改成 `center, columns = None, []`。现有优化器继续使用默认值；同点审计调用该差分公式的两档步长。
4. 在 `gftd_fitting.py` 的 `numerical_check(..., *, stepper=None)` 增加 `capture=None`。当 capture 存在时，在每次原 `_integrate` 返回后立即保存该腿的原状态数组。历史向后保存为 `history_backward_{step}`，历史返回保存为 `history_forward_{step}`，未来保存为 `future_{step}`。将未来原单行 `futures.append(core.observations(_integrate(...), f.mass))` 拆为一条积分、一条可选捕获、一条原观测转换。保留原积分调用的步长、参数、输出网格、次数和门槛。该扩展可按同一端点和方法直接取回六门输入数组，部分失败时已完成的腿仍可保存。

三项公用辅助函数的默认调用及旧配置序列化合同沿用原路径。新模块直接调用 `infer_endpoint`，无须修改旧 `_proxy_candidate`。

## 结果与对账

`fit_state` 给出请求和父模型身份、实际槽位、选择轨迹索引、最终历史目标、充分性诊断和条件信息。公开 `status` 延续 `completed` / `not_identified` 的条件信息语义，后者也可能有合格的最终模型。通过数值资格的父参数可以成为本次拟合输出，条件支持保持未决。部署与后续评分按各自协议。

`fit_private_details.candidates[index].events[local_slot]` 给出两起点终止状态、唯一选点和端点 SHA、六门原数组及各腿哈希。该层的内容哈希见 `private_details_sha256`。未运行的门标 `qualification_status=not_run`。首候选失格在 `FitFailure.diagnostics` 给出 `status=fit_failed`、原记录及已用计数。准备失败标 `preparation_failed`，执行异常标 `execution_unresolved`，由 runner 留存诊断后按协议区分数值失败和需要对账的技术异常。

`counts.actual_residual_calls` 是共享优化调用数；`diagnostic_residual_calls` 单列。 `actual_rhs_calls` 是共享阶段 ROM RHS 合计，可按 `proxy_rhs_calls`、`qualification_rhs_calls`、`diagnostic_rhs_calls`、`information_rhs_calls` 对账。`qualification_suites` 计实际开跑的选点检查。`port_rhs_calls` 属观测驱动端口重建，另列。SVD、条件投影和 NNLS 延用现有计数口径。

当前准备只写了新模块、固定 mock 测试与本说明。科学释放前需接线、源闭包审查与已登记人工资格。本次静态检查不验证运行结果。
