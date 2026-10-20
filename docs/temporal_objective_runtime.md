# 真实区间的时间权重计算

`experiments/temporal_objective_runtime_20261020.py`提供可选的时间目标计算，供现有40维闭合模型联合训练使用。每条自主轨迹仅由100ms历史初始化一次，随后跨不同长度的批次区间持续积分；只计算真实可用的目标点。

`temporal_rollout_errors(model, prepared, initial=None)`返回逐事件的三个量：时间均值与末段均值各占一半的结果、原时间均值、末段均值。真实未来有n个点时，末段取最后ceil(n/5)个点；n=1时总权重仍为1。`PreparedRollout`来自现有`TensorEventStore.prepare`，包含升序活动行和完整真实区间；非法区间在积分前拒绝。输入历史形状为(batch,11,30)，积分步长0.01s，内部状态40维。

`objective(..., tail_weighted=False)`默认使用原时间均值；显式设置为True才使用上述加权结果。事件风险、局部轨迹、隐藏状态一致性、转矩和梯度裁剪沿用现有定义，辅助项保持各自的原历史位置。`one_update`检查目标、梯度和Adam状态是否有限，并在退出或异常时关闭图缓存。调用者仍须遵守既有的优化器、源身份、数据划分与数值资格规则。

对应的纯合成测试可在具备Torch、NumPy、pytest的环境运行：

```powershell
python -m pytest tests/test_temporal_objective_runtime_20261020.py
```

测试覆盖独立逐事件参考、不同真实长度与跨区间末段、值和梯度、原目标Adam等价、辅助项位置、缓存清理与非有限拒绝；CUDA可用时同时执行CUDA用例。旧比较器需要先调用其`load_runtime()`初始化，这已在测试setup中明确处理。

2026-10-20的实际工程验收为34项通过（该总数还包括实验runner和原生监督器测试），设备为CPU及RTX4060Ti CUDA，使用float64。另在私有train资产上完成九次源模型零更新等价和最长14.9s完整反向后的Adam恢复检查。完整记录及首次测试setup失败保存在归档分支`codex/temporal-objective-20261020`的`reports/temporal_objective_20261020_engineering_review.md`与`results/temporal_objective_20261020/`；真实资产留本地，纯合成测试不需要这些资产。

本模块的验收支持计算正确性和可选复用；预测收益、运行加速及科学模型采用仍须各自的实验依据。实验runner、私有执行辅助器、科学端点和默认模型不属于本次工程集成范围。
