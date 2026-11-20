# 正式评价前的接口与结论口径核查

**当前正式评价接口具备完整的冻结输入与选点校验；按以下字段绑定最终已完成回执即可，不需要额外科学计算。** 本次只读 protocol、执行修订、run.py、evaluation/osgkd.py 及已有复核记录，未读取私有数组、运行 ODE 或评价变化中的检查点。以下为新增的最终 action 接口核查，不预判训练结论。

## evaluate_release 必需字段

文件固定为 `results/osgkd_fit297_20261113/evaluate_release.json`，CLI 使用 `evaluate --job-id <新ID>`，不传 `--variant`。

- `released: true`、`run_id: osgkd_fit297_20261113`、`phase: evaluate`、`job_id` 与 CLI 一致、`variant: null`。job_id 为未使用的安全名称；artifacts 和 results 对应目录都必须不存在，否则拒绝覆写。
- `torch_version` 与实际及资格／训练回执相同；`device: cpu`、`threads: 1`，threads 必须是整数。
- `audit_ref` 指向最终 `adopted_for_prediction` 发布回执，完整 `{path,bytes,sha256}`；必须与资格及两份训练回执完全相同。loader 继续验证完整观测采用、物化和 bundle 链。
- `source_hashes` 必须整份复用资格与训练 release 的字典，包括额外条目，不能仅保留必需项。必需项为原 protocol、execution_amendment、port_audit_amendment、架构设计、观测采用设计、run.py、data/osgkd.py、models/osgkd.py、models/osgkd_fitting.py、evaluation/osgkd.py、evaluate_l1_latent.py 的当前 hash。路径必须为规范仓库相对路径。
- `qualification_ref` 绑定 `phase=qualify/status=completed/result.passed=true` 的摘要；`train_refs.C0`、`train_refs.C1` 各绑定 `phase=train/status=completed` 的最终摘要，其 variant、qualification_ref 与本 release 一致。不能绑定 running 回执或中间 trial 文件。
- `selected_model_refs` 的键是 **C0、C1_dynamic**，不是 C1。值逐字等于各 trial 的 `selected.artifact`（无候选时为 null），引用的 checkpoint 身份须有效。C1 静态回退由 `choose_pair` 决定，无需再提交第三个 checkpoint。评价还检查 checkpoint 内 variant 与 attempt。

## 最终结果必须核对

1. **选点和真实完成状态。** 每模型检查 attempts、拒绝、适配原因、实际到达检查点、numeric 资格及内部检查 J；200 不可选，完全相同 J 取更早 attempt。正式评分不得反选。C1 动态与静态回退分列：只有静态 J 严格更低（或动态缺失）才回退；相等保留动态。`C1_score_source` 指定最终 C1 使用哪份成绩，动态评价仍保留。
2. **分母与时域。** 全 train297、稳定267、失稳30及 train 内部两侧分别核对。四个请求时域 1/2/5/8 s 使用请求窗与真实未来交集；`short_record_ids` 不等于删除。无未来或数值未决使该范围 `official=null`，`covered_only_diagnostic` 不能顶替。失稳实际区间单列。所有时间范围明确相对部署或原记录。
3. **指标含义。** `motion_risk_v1` 是事件等权平方误差风险，包含评分时相对运动及共同转速项，不是 RMSE，不能与旧含 p 的 risk 比值。共同转速 RMSE 为 rad/s；端口分别列 system pu torque 的 RMSE 和按 S_m／Se 定义的归一化 RMSE。代码汇总共同转速及端口 RMSE 按实际点／有效分量合并，风险则先对每事件平均，不把两种权重口径混写。正式指标来自细路径，主路径用于同时核对数值资格。
4. **数值资格。** 每范围同时检查有限性、50 状态归一化主细 RMS≤1e-4、max≤1e-3；8 s 失败不抹去已合格短时结果。数值失败不等于真实失稳或模型 LOS。最终稳定积分8 s，失稳积分 max(8,min(15,实际未来))，同一因果初态自主延续。
5. **分类与时机。** 8 s 自主状态和真实覆盖内检出／漏检／误报分列；未知尾段记 uncovered，不能判正确／错误。actual 分类分母只含失稳30及内部两侧；latency_uncovered、numerical_unresolved 分别保留。分类要求全状态数值门及主细 LOS 一致，双触发差≤0.02 s；settled 分歧另报。触发时间误差只比较 `timing_comparable=true` 的共同事件；`predicted_time_record_s` 与加起始延迟后的故障切除时间不得混用。计数列表可能职责重叠，如 missed 与 settled，不将所有列表简单相加当互斥分类。
6. **成本和科学范围。** 单线程 CPU／float64／主细0.005/0.0025 s，执行修订为完整图且重计算 RHS=0。训练、资格、检查、正式评价及序列化／命令耗时分开记；evaluation 内 cost 不含序列化，缺失不填0。单 seed、已暴露 train 的结果是本预算下初步比较，不是独立泛化确认、机制唯一辨识或模型族否证。五项来源诊断仍未决，近似观测合同不回填连续闭合通过。最后分开给证据有效性、优化健康、优化充分性、机制可检验性、效应与确认范围，以及代码／模型／科学决定。

未运行 release 或正式评价，尚无最终性能裁决；上述清单验证静态接口和既定解释范围。

## 补充：极短真实窗口与连续五点 LOS

**连续五点包括已观测初始化历史，不要求部署后另等五个未来点。** protocol 第77行明确拼接11点历史和去掉首点的未来；`src/kiki/evaluation/osgkd.py` 第66、70行实现该拼接。`experiments/evaluate_l1_latent.py` 第30–36行固定阈值和五点，第135、147行调用公开源码 `src/kiki/data/private_dataset.py` 的 `los_index`；后者第40–49行在整条拼接路径上作五点卷积，并返回首个完整五点串的**末点**。helper 的无 h5py 备用分支第139–146行采取相同计数。五个采样点的首末间隔为0.04 s，不应表述为必须有0.05 s未来。

因此，若首个 LOS 末点在原记录0.13 s，其对应串可以是0.09、0.10的观测点，加0.11、0.12、0.13的三个预测点；只要这五点同时满足角跨度和相对速度阈值，就可在0.13 s检出。源摘要的最早真值约0.13 s并不与该判据冲突，也不能由此推出“少于五个未来点必然无法检出”。实际某事件是否满足、是否跨部署时刻续上，只能等冻结 rows 及已保存结果核查，不能从时刻单独判定。

**actual 区间短于五个未来采样点没有单独未覆盖类别。** `osgkd.py` 第125、129–130行用实际可用未来点数调用分类；只要至少一个未来点且预测路径覆盖，便执行。helper 第183–190行只将真值触发≤0.1 s（加1e-9容差）记作 latency_uncovered；第191–218行在有限完整路径未检出时调用 settled 检查。`window_settled` 第163–165行要求至少100个路径点，故总路径短于100点的无LOS有限路径为 `window_no_los_unsettled`，而非稳定证明或短窗未覆盖。

actual 窗口无LOS时，其窗口末点正是实际记录末点，`osgkd.py` 第86–87行的 truth_supported 为真；若数值门合格且标签失稳，汇总第177–191行将其列入 missed_unstable_ids，并同时列入 unsettled_ids。数值门不合格仍是 numerical_unresolved；无未来点才在第129–130行返回无 actual 分类、进 uncovered。8 s 窗口不同：若预测LOS在真实尾段之后，或无LOS且真实记录不足整个8 s，则按第86–87、183–185行记未知尾段未覆盖。

这与 protocol 第77、79行的既定窗口分类一致，静态未发现实现错误。它具有明确的观测窗口限制：若阈值连续串在实际尾段结束前未凑齐，actual 窗内不会触发；但不能把该限制改写成统一的“五个未来点检测下限”，也不能结果后把这些既定漏检改成未覆盖。最终可说明其真实窗口和判据含义，事件数量及预测触发情况等冻结 rows 再核对。
