# 匹配预测 runner 实施复审 v2

结论：v1 三项阻断已解决，接受 v2 进入登记的元数据准备和首次固定资格。尚未执行资格，不能据本报告直接释放共享适配、部署或真实评分。原协议、候选与样本范围、各阶段调用上限及真实未来门均未改变。

本次对照 v1 归档原字节与 v2 全部差异，复读资格驱动、增补 mock 和相应公共配置/存取路径，并作标准库 AST 解析。没有导入被审模块、运行测试、模型、ODE、证书或读取原始数据。

## 冻结身份

| 文件 | SHA-256 |
|---|---|
| `experiments/gftd_matched_prediction_20261109/run.py` | `5cad913345658b842b270c50e1313b64607c5a543920348f783b03c2fd08f352` |
| `experiments/gftd_matched_prediction_20261109/qualify.py` | `2b3129ee9813f20621e52d55809966e0ac33ab872a9911ad6db9437f9b60c73b` |
| `tests/test_gftd_matched_runner.py` | `9aafbf2ec10e60cb66b6f8c3ec87297da766f3a8d56c3b6ad7c9501b4a786288` |
| `runner_integration_notes.md` | `7f51b2dc14ebc14c8e04aac58c58b338d0f47ae997ec856cf5858533c240142d` |

四份实际哈希与冻结值一致，`implementation_v1/` 下对应四份归档也与 v1 哈希逐项一致。[v1 审查](runner_implementation_review.md) SHA 为 `9ea1b9378b287d2371705ba41b5964dd6446cac7997e6134390f2cc67922c491`，原阻断记录保持不变。

本复审继续绑定协议 `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43`、设计审查 `8322c6791a192b9bcd0e881d7b53078527d3f17ccad61f1b14fdcd8562972b5b`、算术澄清 `c73f38d66c56fbaef2cf8bc4fce9ea7a978fe72ab56cf778778ee291842582d8`、隔离分析实施审查 `9c360b0b7fdda819765fb70512b0441e426d39184bcff682ec5da59bd8039cc3`，以及共享适配/公共接线复审 `c3b1c5276e8f03515ac951a6ab634933dc32729a3d17194c7dedf586b36899c0`。

## 阻断闭合

R1 已解决：实际 RHS/H 的已登记调用路径由类方法和已加载函数别名共同包装，在调用前检查原累计额度并递增独立账本。profile 只观察实际入口、其他代数函数及禁止入口，不再从 profile 抛异常。profile 丢失或发现未计量入口后，状态保持成本未知，后续受控 RHS/H 入口拒绝；重新出现 profile 不会自动清除此状态。收尾比较实际入口观察数与独立账本，任何不一致、违规或未知成本都不能通过资格或自动取得下一次额度。当前固定测试的真实模型调用路径均通过这些受控入口。

R2 已解决：FitFailure 只有明确 `cost_unknown=false` 且登记计数对账成功，才可能作为普通数值失败完成。缺字段或为 true 时保存原诊断和已知成本，Job 保持 interrupted；阶段对账只把显式成本已知的 completed/fit_failed 计为终态。下游阶段另外要求完整实际成本和逐 Job 成本状态闭合。L3 调用前保存 `l3_started=true`，返回有效 counters 后才填入实际 RHS/梯度/eig；异常未返回时 `l3_counts=null`，不再以预置零计数代替未知。正常成功的共享模块路径保留完整聚合账本，其终态来源已由共享模块复审接受。

R3 已解决：评分显式源码列表加入 `src/kiki/data/private_dataset.py`，覆盖原评分器经 `train_joint_latent.py`、`run_private_m2.py` 的间接导入。该改动只扩充源码身份；没有提前加载原事件、父模型或真实未来。评分的原纯数值调用范围保持不变。

## 新增 mock 的范围与账本隔离

资格计量测试创建自己的 `QualificationMeter`、`Counter` 和 `mock_call`。它模拟已用 127 次、额度拒绝、禁止入口、profile 丢失及未包装入口；这些数字只存在于局部 mock 账本，不写入实际资格 meter。测试没有调用真实 RHS/H、真实求解器或优化器。外层 profile 可以记录对应 fixture 函数次数，不能把这些次数当作模型调用。实际资格回执仍须分别报告模型、代数和 mock 类别。

未知拟合成本测试以替代 model.fit 抛出预设诊断；12 次 RHS 数字只是回执 fixture，不发生 12 次实际求值。L3 异常测试替代证书与 rom_modes，不执行证书或谱计算。两者检验 partial/null 保存及阻止冻结，未扩充人工科学工作。

公开 fit 测试现在分别覆盖静态 exp 和有界核。共享适配本体被 mock，返回按公共 make_config/固定 DP8 合同生成的可验证配置，保留白名单、参数副本和互斥检查。save/load 后核对 v2/v3 核身份、functional 指纹、inference_config、fit_state，并复存为相同字节。部署内层兼容字段 `proxy_start_calls=200` 与本轮共享策略 1200 分别保存。上述配置、指纹和存取路径不调用模型 RHS/H、ODE 或真实优化；其两种核覆盖属于已登记的接口资格，没有增加人工物理点。

## 适用决定

v1 已接受的其余边界继续成立：原297分母、固定24索引映射、各阶段唯一 claim、两模型隔离与部署并行、六门原数组捕获、同一正式端点的完整主细预测、严格参考与原资格分离、一次 L2/L3，以及冻结全部选择后才读取真实未来。

代码决定为接受本版 runner 进行固定资格；模型与科学决定保持未执行。下一步可以由 root 按既有授权生成绑定本复审的 prepare/release，再执行首次固定资格。只有其实际回执通过、成本已知且源身份一致，后续科学阶段才可按各自唯一释放执行。本次没有新增科学授权或扩大测试、数据、调用预算。
