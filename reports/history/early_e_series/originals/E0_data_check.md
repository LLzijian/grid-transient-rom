# E0 技术报告:数据通路移植与校验

> 历史保留标记（2026-09-22）：本报告仅记录早期数据管线/QA与开发基线结果。
> 当前状态见 [current_state](../docs/current_state.md)；结论边界以 [0010 审计](accepted/0010_project_conclusion_audit.md) 为准。

日期: 2026-09-04 · 状态: **PASS** · 证据: `results/e00_data_check/summary.json`

## 1. 目标

在 kiki 中重建 kron-reduce 的数据通路,保证:同一份 canonical 数据、同一份
line-disjoint 划分、同一份因果输入契约,全部逐值一致。这是后续所有模型
数值结论的正确性地基。

## 2. 移植内容(verbatim 复制 + 导入重写,零逻辑改动)

| kiki 文件 | 来源 (kron-reduce) | 职责 |
|---|---|---|
| `data/store.py` | `data/store.py` | `CanonicalStore`:HDF5 懒加载,315 事件索引,时间契约校验 |
| `data/schema.py` | `data/schema.py` | `EventSequence / EventMetadata / CanonicalSplit / DatasetMetadata` |
| `data/contracts.py` | `data/contracts.py` | `InputContract`(允许/禁止字段白名单)、`ModelDataRequest`、ZOH/OBSERVED |
| `data/splits.py` | `data/splits.py` | `load_split`:视图-划分绑定 + train/val/test 泄漏检查 |
| `data/adapter.py` | `data/adapter.py` | `adapt_event`:事件→模型输入(COI 重投影、ZOH 调度、契约断言) |
| `physics/{constants,coordinates,units,graph,aggregation}.py` | 同名 | COI 投影、有效质量 m=M/ωs、完全图关联矩阵、聚合(备用) |
| `models/base.py` | `models/base.py` | `DynamicsModel` 接口、`RunContext`(seed=324) |
| `models/integration.py` | `models/integration.py` | 共享 COI 保持 RK4(最大内步长 0.01 s,ZOH/线性 Pm) |

**有意删减**(相对旧项目):`batch.py / stratification.py / audit.py /
p1_*.py`(数据层),`coherency / graph_quotient / global_quotient /
multiresolution`(物理层,聚类路线已被 AFSD v5 `FULL_RESOLUTION_REQUIRED`
否定),torch 版 RK4 暂未移植(E7 基线时再补)。

## 3. 校验结果

数据集:`E:/03-nn_Transfer_Func/kron-reduce/data/ieee39_postclear_canonical_v1`
(只读引用,不复制)。

| 检查 | 结果 |
|---|---|
| 事件总数 / 稳定 / 不稳定 | 315 / 264 / 51,与 schema 一致 ✅ |
| 5 折划分规模 | F1: 209/43/63;F2: 214/37/64;F3: 212/42/61;F4: 206/43/66;F5: 207/47/61 ✅ |
| **全部 5 折 fault-line 不重叠**(test 线路 ∩ train∪val 线路 = ∅) | ✅ |
| F1 train+val 252 事件全部通过因果契约适配 | 252/252 ✅ |
| COI 状态残差 max \|Σ m·x\| | 1.33e-15 ✅ |
| COI 功率残差 max \|Σ Pm\| | 7.11e-15 ✅ |
| 非有限目标 | 0 ✅ |
| ZOH 调度恒等于 `Pm_clear`;`delta0 == delta[0]`;`t[0] == 0` | 全部逐值断言 ✅ |

## 4. 结论

数据通路与旧项目逐值一致(COI 残差在双精度噪声量级)。kiki 的表示固定为
`generator_10`(10 机全分辨率),主 Pm 策略为 ZOH at `Pm_clear`。
可以进入 E1(M0 基座复现)。

## 5. 与旧项目的已知差异(有意为之)

- 表示层默认 `generator_10` 而非 `clustered_7`(v5 定论;7 群聚合代码保留备用)
- 无 project_harness 治理层;不可变 run 目录纪律保留(`results/<stage>/`
  已存在则拒绝覆写)
- 稳定性评估将使用统一固定 H=100 因果 rollout 派生标签(旧项目教训:真实
  `valid_length` 导致 FSR=1.0 不可辨识)
