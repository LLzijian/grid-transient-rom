# 离线电气功率替换：电气闭合与源方程余项

日期：2026-09-07。冻结代码：`8c6e1bd`。决定：接受诊断证据，不晋升预测器。

## 方程定义

源生成脚本读取 `GENROU.tm` 并最终在 canonical 中命名为 Pm，读取 `GENROU.Pe`
作为离线 Pe。canonical 对两者做零总功率投影，对速度做惯量 COI 投影并乘基频。
这些名称不意味着 Pe 就是转速方程中的电磁转矩。

[ANDES 2.0 GENBase 源码](https://raw.githubusercontent.com/CURENT/andes/v2.0.0/andes/models/synchronous/genbase.py)
给出 `M*d(omega_pu)/dt = tm - te - D*(omega_pu-1)`。
[GENROU 使用 Flux0](https://raw.githubusercontent.com/CURENT/andes/v2.0.0/andes/models/synchronous/genrou.py)。
由其磁链、电流及端口功率定义可推导，在设备在线时
`te = Pe + ra*(Id**2 + Iq**2)`。
因此本轮积分 `Pi(Pm-Pe)/(M/omega_s)` 后留下的速度余项包含电阻损耗、
阻尼、采样积分误差，以及可能存在的源参数/转换不一致，不能直接称为数据错误。
COI 数据还去掉了共同转速；一般情况下，物理阻尼不能仅用相对转速精确重建。

本地源证据：

- `E:/03-nn_Transfer_Func/swing_equation/external_sim/andes_ieee39_generate_dataset.py`
  的 `extract_and_check` 读取 Pe 与 tm。
- `E:/03-nn_Transfer_Func/swing_equation/scratch/one_off/2026-07-22-discovery-cleanup/generate_ieee39_near_critical_unstable_cases.py`
  对 dense 源同样读取并插值 Pe 与 tm。
- `E:/03-nn_Transfer_Func/swing_equation/discovery/lib/data_graph_sindy.py`
  对两者做 COI 功率投影，canonical builder 保存转换后的序列。

源代码用于解释字段；尚未重跑源仿真，也未验证所有场景的真实 ra/D 参数。

## 冻结实验与结果

F1/F5 全部 90 个验证事件，各取前 min(101,T) 个真实采样点。
73 个完整 H100 事件与 e14 配对；17 个短不稳定事件单独报告，不填补、不外推。
未读取 test payload，不拟合参数、不筛选种子。采用梯形积分，计算
`omega(t)-omega(0)-integral(Pi(Pm-Pe)/mass dt)` 的时空 RMSE。
这不是自由滚动预测分数，也不是改进模型的性能上界。

| 范围 | 事件数 | 稳定数 | 速度余项 RMSE 事件中位数 (rad/s) | 最大值 |
|---|---:|---:|---:|---:|
| F1 H100 | 34 | 34 | 0.303519 | 0.786142 |
| F5 H100 | 39 | 39 | 0.306860 | 0.538330 |
| F1 短路径 | 9 | 0 | 0.203418 | 0.353237 |
| F5 短路径 | 8 | 0 | 0.093875 | 0.167053 |

完整 H100 的现模型沿真值、真实 Pm、稳态隐初值残差中位数为
F1 0.205589、F5 1.261414 rad/s。Pe 替换仅在 F1 14/34 个事件改善，
F5 则为 30/39；对零隐初值比较分别为 14/34、28/39。
Pe 对照没有学得的有效阻尼与隐状态端口，故它与现模型不是单因素、等结构比较。

F5 最差事件 `ncunstable_dense_0112`：现模型稳态/零初值残差
4.78935/3.83262 rad/s，Pe 替换余项 0.53833 rad/s。
这显示测得的电气信息对该事件有很大解释价值，但不能把差值唯一归因于
某一个高阶电气状态，也不能把 Pe 当作因果模型输入。

两折 highorder 子组的余项中位数约 0.301 rad/s，dense 子组为
0.517（F1）和 0.431（F5）。没有时间趋势比较；短路径误差更小也不能表示
不稳定事件更易预测，因为其积分时长不同。最大总功率不平衡仅 2.13e-14。

## 下一模型实验的方向

优先探索单开发折、固定种子的“真实训练 Pm 驱动的电气闭合 + 因果 Pm 动态”
解耦拟合，避免让电气系数补偿训练中被 ZOH 删去的输入变化。部署从清故障初值
积分各模块，不使用未来 Pm/Pe。电气状态需通过总储能中的耦合项交换能量，
而不是把所有缺项都强制当作瞬时耗散。具体实现仍需独立的精度、能量恒等式、
初值可用性与积分收敛验证，本轮没有证明这个方案会成功。

## 核验与限制

复现脚本 `experiments/diagnose_electrical_balance.py`；沿用项目脚本及 Git
报告作为可检查证据，不另建 notebook/展示界面，以保持用户要求的项目简洁。

输出 `results/e15_electrical_balance/summary.json`，SHA256
`3e61734d9576f55a150f473e1ebf3fd88706f4486c3a57c0b74a9716b78cad94`。
记录 schema/e14 哈希、数据 build、Git、Python/NumPy 版本、事件签名，另对每个
Pe 切片取 float64 字节哈希（旧事件签名不含 Pe）。

90 个唯一验证 ID、长度、事件签名、Pe 切片哈希、依赖哈希与有限 JSON 均通过。
独立采用 SciPy cumulative_trapezoid 及“先除质量再投影”的等价计算，
重现全部余项 RMSE 和终端冲量；重新核对各组计数、中位数及最大值。
全套 41 测试、相关 Ruff、compileall 通过。

结论可信度：可附限制分享。未验证源方程的完整闭合，未修改数据、生产模型、
论文正文或 PDF，也没有新增稳定证书或模态精度结论。
