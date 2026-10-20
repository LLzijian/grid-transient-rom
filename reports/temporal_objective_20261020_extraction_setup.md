# O/F/T结果提取准备

2026-10-20。新增`experiments/extract_temporal_objective_20261020.py`，将同一次端点评价保存的JSON转换为紧凑指标。固定读取`temporal_objective01/evaluation/{seed}_{O|F|T}/step000512/queue`的九组证据，先验证全部完成标记和SHA256，再读取指标。原366、新19及共同385按照已冻结`cohort_identity.json`的完整身份集合划分，保留342条稳定事件和43条失稳样本；不按事件名称猜测归属。

原1/2/5/8 s点误差、稳定5 s风险、共同频率rad/s与机械功率100 MVA基准pu/MW、实际区间至15 s误差及分类均按既有定义汇总。物理RMSE合并SSE及实际标量点数；8 s和实际15 s数值资格独立，任何相关事件未决即保留对应总体指标为null。F−O、T−F、T−O输出逐项差值，不作显著性或候选采用判断。

runner已经保存`extras.temporal`中的实际点数、末20%点数、末段点误差和终点分组，因此无需再次推理。提取器同时要求同事件实际15 s完整数值资格；finite本身不能使末段指标合格。实际未来点数`n≤500`与`n>500`两组分母及分类全部输出，并核对九模型的真实终点一致。缺少末段事实时明确输出事件ID及未决值，禁止重算模型补齐。事件级记录写入本地`temporal_objective01/extracted_event_rows_step000512.json`。

合成测试使用stdlib unittest，不依赖私有资产，覆盖固定集合与同前缀事件、缺失及重复ID拒绝、风险算术、物理单位和点数、数值资格传播、末段事实缺失以及500/501边界。五项通过，测试自身0.174 s；原生命令2.4239273 s。本次没有读取新训练或评价成绩，也没有运行实际提取。正式调用须等待九组评价完成，由root核对路径与费用后执行：

```powershell
& 'C:/Users/LLLzijian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' experiments/extract_temporal_objective_20261020.py --output NEW_JSON
```

本文件只记录提取准备；科学解释和是否满足原门槛由root依据实际完整结果决定。
