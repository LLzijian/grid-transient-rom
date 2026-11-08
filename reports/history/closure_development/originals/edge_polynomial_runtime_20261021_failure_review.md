# 多项式运行后端 QA02 失败解释审查

结论：本轮未通过已登记的全部检查，应按失败关闭。四项失败不能直接解释为多项式展开的数学错误；源码显示测试只修改了同一物理量的部分副本，旧通用路径与新快速路径随后读取了不同状态。当前证据也不足以宣称后端在任意非单位惯性、功率尺度或直接修改相角缓存后保持等价。

## 失败定位

- `tests/test_edge_polynomial_runtime_20261021.py:93–103` 同时改变包装模型的 `mass` 和 `base.p_scale`，没有同步 `base.structured.mass`。旧 `TrainingClosureROM.field` 的通用分支（`experiments/closure_training_runtime.py:109–112` 附近）先通过 `interface_to_state` 将速度乘外层惯性，再交给 `StructuredROM.energy_and_gradient`；该函数在 `experiments/structured_rom.py:123–128` 使用其自身惯性计算动量梯度。新快速路径在 `experiments/edge_polynomial_runtime_20261021.py:138–154` 直接使用速度作为这一梯度，依赖原有内外惯性一致条件。pytest 的角度导数第9项分别约 .0177 与 .0407，差 .023，比例对应被改成 2.3 的惯性，符合这一不一致的表现。此项同时修改两个量并在首个 field 断言处停止，因此不能单独定位 p_scale 错误，也没有执行后续 RK4 断言。
- `tests/test_edge_polynomial_runtime_20261021.py:127–128` 只将 `base.phi` 清零。`LocalMemoryROM` 在 `experiments/local_memory_rom_0098.py:199–205` 分别建立外层 phi 和内部 StructuredROM；后者在 `experiments/structured_rom.py:82` 附近自行计算 phi。旧通用能量路径在 `experiments/symbolic_closure_rom.py:549` 读取内部 StructuredROM 的基底能量/梯度，新增闭合项另读外层 phi；新快速路径的基底与闭合均读外层 phi。于是测试制造了两种基底相角，而非在两个一致的物理配置上比较算法。三种表达式失败的最大绝对差完全相同，均为 2.1647673324501696e-05，且均只出现在30/120项；这与共同基底项不一致吻合，不能据此认定 c 中心化或四次项编译各自出错。

上述判断为源码与已保存失败数值相互印证的静态归因，未读取张量存储地址或进行新的数值验证。适配器独立重建模型且复制 state_dict；它没有承诺任意原位修改后保持对象内部各物理副本自动联动。测试暴露了这一可变配置接口条件，不应把测试失败删除或改记为通过。

## 能保留的证据与边界

`private_data/closure_mechanism_20261019/P0/edge_polynomial_runtime01/qa02/pytest_stdout.txt` 记载 26 项通过、4 项失败。六个冻结池候选在原一致配置上的完整双步长5 s预测、向量场及梯度检查通过；完整0.5 s梯度、实际触发裁剪后的 Adam，以及新后端32步恢复到34步检查也通过。三项相角测试在清零之前的比较与端点反向比较已通过，之后才失败。这些是限定配置的正面工程证据，不能覆盖未通过的强制检查。

本轮没有运行性能基准或正式科学试验，因而没有速度收益、完整费用可容纳性、未知表达式恢复或科学模型采用的证据。保持本轮关闭，不再修复、重试或用上述归因放宽门槛；后续问题与研究范围由 root/Astra 独立决定。

## 审查操作

仅阅读源码和 QA02 已保存 stdout；未执行 Torch、测试、推理、数据读取、Git 或新增试验，未修改代码。工具 wall 时间最终值由审查消息报告（此前四次只读合计 10.7763135 秒；本次读取及报告写入另计），计既有 P2review180。
