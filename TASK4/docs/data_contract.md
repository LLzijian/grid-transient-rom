# 与KIKI的只读数据契约

## 已按上传源码核对的默认布局

```text
parent/
  kiki/
    private_data/
      ieee39_postfault/{dataset.h5,dataset_index.csv,splits.json}
      ieee39_expansion/{dataset.h5,dataset_index.csv,splits.json}
      task2_m3_0020/model.npz
    configs/private_dataset.json
  TASK4/
```

`configs/project.json`中只存相对于kiki的路径；默认根`../kiki`。大小写按真实文件系统核对，可用环境变量 `TASK4_KIKI_ROOT` 显式覆盖。路径错误不是理由去全盘扫描HDF5事件。

## 数据字段与坐标

来源：已提供Loop0101的 `experiments/train_local_memory_0098.py`、`src/kiki/data/private_dataset.py` 和 `private_pm.py/private_rom.py`。

- 索引：CSV `event_id,fault_bus,label,trajectory_end_s`；分割JSON有train/validation/test。
- HDF5共享数据：`inertia`、属性`omega_s`、`equilibrium/delta,omega,Pm,Pe`；适配器只读取实际需要的公共字段。
- 事件数组：t/delta/omega/Pm/te；omega原始为绝对pu，te和Pm沿用系统pu。
- canonical delta = raw_delta − equilibrium_delta。
- canonical omega = (raw_omega − 1) × omega_s，**保留共同频率**。
- p = raw_Pm − baseline.pm_star；mass = inertia / omega_s。
- 初态取索引10，history包含0..10，共11点。第一项预测误差从索引11开始。
- 当前转矩辅助真值 = te − pm_star；模型对齐量 = p − pi_dot。这个量是沿用KIKI约定的总反作用，不擅自命名为单独Pe。
- 公共基线delta_star可能做过COI平移，只检查边角差一致，不要求绝对角完全相同。

TRAIN369原始/346合格/35失稳；validation117原始/107合格/12失稳是当前源报告的**预期核验**，不是本项目已重新读取所得。若用户KIKI已更新，先核实源变更和版本，不删事件凑数。阈值只用于数据版本检查，可在有依据时前瞻更新。

## 禁止行为

禁止读TEST载荷；禁止枚举所有events后筛TRAIN；禁止用全数据归一化；禁止把未合格边界数据混入；禁止填充失稳尾部；禁止在KIKI中生成缓存或改基线。元数据允许看到test ID以检查隔离，不等于获准读取test轨迹。

## 源只读的实现范围

`data.py`采用h5py.File(...,'r')并只索引名单。输出写入TASK4路径并拒绝..或符号链接逃逸。不会import或运行KIKI脚本。代码护栏不是OS沙箱，执行代理仍负有遵守只读权限的义务。

## 数值评价与原有判据

当前独立评分器实现真实区间的COI/common/p误差和两步长核验，保留全部未决ID；不把未决事件删掉后重算全体指标。2026-09-30已移植KIKI的LOS/window_settled完整分类流程，并通过合成回归与独立只读复核；真实冻结端点评价仍待执行。

移植来源为kiki的 `experiments/evaluate_l1_latent.py` 及其调用链，保留原阈值、持续时间、观测前缀拼接、绝对/切除后/决策后时间及数值门作用范围。轨迹1/2/5/8秒均从100ms决策状态计算未来时长；完整分类拼接前10个已观测样本。不能用仅到真值截断的预测冒充完整固定时域失步判别。原LOS的RMS/触发时差数值门与轨迹误差的分量最大差门分别报告，不互换。

## 缺数据时

本包不携带真实KIKI数据、模型权重或冻结TEST。在目标机器把文件夹与已有kiki并列即可。若缺M3模型/HDF5，报BLOCKED_DATA，指出准确缺失路径；可继续无数据实现/测试，但不能伪造数据或把合成烟测当研究结果。
