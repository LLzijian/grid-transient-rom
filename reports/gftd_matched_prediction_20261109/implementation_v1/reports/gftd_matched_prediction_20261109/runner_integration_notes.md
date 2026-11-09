# 匹配预测运行接线

匹配预测使用五份独立释放依次核对前置证据、运行人工资格、冻结两模型、冻结全部初始化选择并评分。两模型分别从已登记的父文件调用公开 `fit`。部署使用各自最终模型，真实未来由最后的评分阶段打开。

## 接线和释放

闭合并归档有界初始化 Run 后，按[共享适配接线说明](fit_integration_notes.md)接通 `RunContext.gftd_matched_prediction_v1` 及三个默认兼容的辅助参数。公开 `fit` 分派须传入 functional 副本、严格白名单验证后的包、模式及登记请求，保留原 `FitResult` 和存取流程。runner 的 `fit_job` 检查最终模型核、非边常数、尺度、24 个槽位、诊断哈希及实际计数。

所有释放 JSON 都含 `run="gftd_matched_prediction_20261109"`、`released=true`、唯一 `action_id`、`phase` 和 `qualification_attempt`。文件引用形如 `{"path":"仓库相对路径","sha256":"字节哈希","bytes":123}`。首次资格的 attempt 为 0，文件名无后缀。最多两次明确缺陷的复查分别用 attempt 1 和 2，资格、准备及各自 claim 文件名加 `_1` 或 `_2`。

| phase | 额外释放字段 | 产生的关键回执 |
|---|---|---|
| `prepare` | `prerequisites` 含 `current_closeout`、`current_review`、`current_source_archive`、`current_execution`、`implementation_review`、`qualification_clarification`，值均为文件引用。复查另含 `technical_fix` | `prepare.json`，含 26 份已归档旧源码的验证、实际新源闭包、依赖版本、登记资产身份和 24 槽位映射 |
| `qualification` | `prepare_sha256`、`previous`。首次 `previous=[]`，复查按顺序列出此前各资格回执引用 | `qualification.json`，含固定测试、实际调用、共享上限、违规及已知成本 |
| `fit` | `prepare_sha256`、`qualification_sha256`、`workers`，两个独立模型可用 2 | `fit/execution.json` 和 `fit/model_freeze.json` |
| `deploy` | 与拟合同一资格绑定、`workers`、`fit_execution_sha256`、`model_freeze_sha256`、`fom_sha256`、`kron_sha256` | `deploy/execution.json`、`deploy/initialization_freeze.json`、L2/L3 回执 |
| `score` | 与拟合同一资格绑定、`workers=1`、`fit_execution_sha256`、`model_freeze_sha256`、`deploy_execution_sha256`、`initialization_freeze_sha256`、`future_sha256` | `score/execution.json`、逐事件结果和 `score/l1_summary.json` |

命令在工作区根目录执行。

```powershell
.venv-gpu/Scripts/python.exe -B experiments/gftd_matched_prediction_20261109/run.py --prepare <prepare_release.json>
.venv-gpu/Scripts/python.exe -B experiments/gftd_matched_prediction_20261109/qualify.py --release <qualification_release.json>
.venv-gpu/Scripts/python.exe -B experiments/gftd_matched_prediction_20261109/run.py --release <stage_release.json>
```

结果位于 `results/gftd_matched_prediction_20261109/`。准备核对有界初始化全量回执完成 272 新包并复用 25 小试，以及旧准备清单中 26 份源文件的归档原字节。准备不打开父 checkpoint、合法包或真实未来。拟合开始才读取父文件和合法包，核验两父共用的冻结常数和尺度。资格复查共用 128 RHS 与 2000 H，已有通过或成本未知的回执会阻止自动复查。

## 运行、冻结与对账

每阶段有独占 `started.json`，每个拟合、部署及分析 Job 有独占 claim。Job 的中间回执保持 `running`，已知成本可用于对账。终态 `completed` 和已登记的 `fit_failed` 才进入阶段完成表。阶段失败保留已完成、部分、未知和未领取四类 Job，未知成本写 `null`。同一阶段的 claim 会阻止重放。

部署时每个可用最终模型按原顺序处理 297 包。逐包回执区分原六门、严格 Radau 参考和长时主细 DP8。六门全状态数组在原调用时捕获，各腿绑定起点字节。15 s 的主、细轨迹包含各自内部网格，并绑定同一正式端点。拟合失败模型在初始化冻结表保留 297 行 `not_run`。L2/L3 读取相同的最终模型和正式点，分别保留模型证书、逐点包络及模态匹配。

评分启动前核对两个最终模型、完整初始化选择表和逐 Job 证据的哈希。评分器在这一阶段才加载旧 L1 纯数值入口，验证已登记的 train 缓存、原事件顺序、267 稳定与 30 失稳标签及每包合法历史的对应关系。缓存只解码 `metadata_json` 和 `event_XXXX_x`。评分记录保留原分母和失败行，主、细预测的正式初态字节一致。原始拟合诊断、模型、初始化与预测数组留在 `artifacts/gftd_matched_prediction_20261109/`。

只读巡查可读取当前阶段的 `execution.json.status`，按逐 Job 回执的终态计数并用 claim 的 PID 核对工作进程。长作业的阶段汇总在收尾时写入。已知工时、实际 RHS 和失败成本来自逐 Job 回执。

本次准备的核验限于静态源码与 AST。公开分派、实际计数、存取和评分边界由冻结后的固定资格检验确认。
