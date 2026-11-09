# GFTD 固定初态诊断独立结果复核

**接受诊断判读及分支 A 进入完整初始化小试。** 固定初态比较确认 23 个旧失败点存在未来 RK4 离散误差，14 个调制方案溢出点呈现连续幅值增长。固定 DP8 在原六门下使 25 个旧失败点通过，5 个原成功输出和另外 5 个 stage1 对照均通过。判读适用于这批预先选定的状态。 [Scientist 判读](diagnostic_assessment.md) · [固定初态执行回执](../../results/gftd_initialization_precision_20261109/fixed_state/execution.json)

## 同一冻结状态的两类证据

- **离散误差。** 从保存轨迹复算了 49 个状态的投影差、参考资格与可计算收敛阶。历史参考 49/49 合格，未来参考 33/49 合格。其中 8 个 static、15 个 modulated 旧失败点的原 1 s 未来 RMS 差超过 1e-5，RK4 相邻收敛阶约为 4。最细 RK4 对严格 Radau 的未来投影 RMS 最大值分别为 1.554e-8、1.673e-7。已有 48 个 static 和 129 个 modulated 非负时刻谱样本中，沿正向衰减的模态均未超出原步长 RK4 稳定域。这项谱判断限于已采样时刻。 [逐状态摘要](../../results/gftd_initialization_precision_20261109/diagnostic_digest.json)
- **幅值增长。** 14 个溢出样本在原未来主步长下的失败步号均与旧记录一致。用保存的失败子步状态和同一模型系数，复算四档 RK4 共 56 个失败子步的指数、势能及活跃标记。每个失败子步的最大活跃指数超过 float64 上限，且该边 `beta*psi<0`。原主步长下最大活跃指数的最小值为 970.83，最大非活跃指数的最小值为 486.12。即使略过非活跃零项，这些已保存子步仍有活跃指数溢出。三参考在共同 0.01 s 网格的末点为 0.32–0.98 s，全状态除以 0.1 后两两最大差为 2.170e-7。幅值从零增长至共同末点的最大绝对值 3.015–5.761。完整 1 s 参考仍未决，证据支持共同时间段的连续增长，有限时爆破及真实系统失稳未获证明。 [溢出和共同网格证据](../../results/gftd_initialization_precision_20261109/diagnostic_digest.json)
- **单参考到限。** modulated 索引 074、178 的普通 Radau 和 DOP853 均完成 1 s，严格 Radau 达到 20000 次 RHS 上限。这两点的三参考资格保持未决，单列于 14 个连续增长样本。 [参考资格摘要](../../results/gftd_initialization_precision_20261109/diagnostic_digest.json)

## 固定 DP8 的通过范围和成本

复算 49 个固定状态的可用原门值。旧失败转合格为 static 8、modulated 17。后者包括两个三参考未决的样本，因此 25 个新增六门通过点中有 23 个已有诊断三参考证据。剩余 14 个旧失败点均为原溢出组。五个 stage1 对照单列。结果满足完整初始化小试的前置条件。全体覆盖、预测效应和共享参数的因果作用仍待各自的证据，原覆盖为 static 46/297、modulated 1/297。 [固定初态完整性回执](../../results/gftd_initialization_precision_20261109/fixed_state_integrity.json)

| 阶段 | 状态数 | 积分尝试 | 积分完成 | 积分 RHS | 采样及 Jacobian RHS |
|---|---:|---:|---:|---:|---:|
| 固定状态诊断 | 49 | 882 | 782 | 1,950,911 | 45,426 |
| 固定 DP8 小试 | 49 | 294 | 266 | 302,115 | 不适用 |

诊断 digest 的 105 项输入哈希全部吻合，诊断与固定小试执行回执分别与阶段完整性记录一致。审查读取了已生成数组与冻结模型系数，未调用 ODE、Jacobian 或谱求解，未读取真实未来。 [诊断完整性回执](../../results/gftd_initialization_precision_20261109/diagnostic_integrity.json) · [固定小试完整性回执](../../results/gftd_initialization_precision_20261109/fixed_state_integrity.json)
