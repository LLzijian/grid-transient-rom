# 物理空间极值监督

`experiments/spatial_extrema_runtime_20261020.py`提供可选的角差范围与最大相对转速监督，供现有40维闭合模型联合训练使用。原逐机轨迹、共同频率及功率目标保留。

角差范围是同步角偏差加回各机平衡角后的最大值减最小值，单位rad；最大相对转速是惯性加权共同转速消去后的绝对值最大值，单位rad/s。平衡角必须以显式tensor传入，形状为10或(batch,10)，与惯性系数和机器顺序一致；不增加模型参数或部署观测。角度保持原连续表示，不作2π折返。

`extrema_error(pred, truth, mass, equilibrium_delta)`返回组合误差与两个归一化平方误差分项。角差除以固定`2*METRIC[0]`，速度除以`METRIC[1]`；组合误差为两分项均值。使用amax/amin的并列梯度分摊。

`objective(..., equilibrium_delta, spatial_weighted=False)`默认与原完整目标等价；True时在原目标上增加10倍逐事件空间误差风险。两个目标共用同一次自主长轨迹，只在真实可用点上取每事件均值；局部轨迹、隐藏状态一致性和转矩保持各自原位置。`one_update`沿用原Adam更新、梯度裁剪及有限性/缓存清理检查。该模块固定0.01s步长及现有11×30历史接口；调用者仍须执行原科学协议和数值资格检查。

纯合成测试不需要私有数据：

```powershell
python -m pytest tests/test_spatial_extrema_runtime_20261020.py
```

2026-10-20的整套工程QA实际30项通过（总数含归档的runner和原生接口检查），设备为CPU与RTX4060Ti CUDA。runtime测试覆盖独立手算、真实变长区间、物理平衡角、共同角平移、机器重排、并列梯度和有限差分，以及原目标组件/全部梯度/Adam等价、辅助位置和非有限拒绝。首次QA的计数张量类型错误与唯一一行修复完整保留，测试输入、断言及容差未改。另在私有train资产上验证六种源初始化与最长5s完整反向后的Adam恢复匹配。

记录位于归档分支`codex/spatial-extrema-20261020`的`reports/spatial_extrema_20261020_qa.md`及`results/spatial_extrema_20261020/`。真实资产留本地；实验runner、私有监督器、科学端点和默认模型不属于此工程模块的集成范围。计算正确性通过不代表预测收益、运行加速或模型采用。
