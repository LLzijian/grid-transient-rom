# 非负四次记忆储能计算模块

`experiments/quartic_memory_runtime_20261020.py`在现有七项模型上提供一个可选节点储能项`a*sum(q**4)`，`a>=0`，解析梯度为`4*a*q**3`。q沿用当前隐藏坐标。模块保持十机40状态、100ms初始化器与既有交换、耗散和端口方向。

`build_variant(source, 'N')`复制原七项结构；`build_variant(source, 'Q')`增加零系数四次项，保留原参数。a位于`coefficients`向量，通过表达式身份查找，使用closure学习率。`one_update`使用原训练目标，按实际参数集合裁剪梯度，Adam更新后先检查有限性，再只将a投影到非负范围，保留Adam矩。返回值包含投影前后值、梯度与更新量。

每个优化图缓存仍连接梯度的a张量；结束、异常、加载或迁移后清理缓存。导出要求图已关闭且a有限非负，恢复核对两个系数副本。通用符号计算与NumPy解释器均保留四次表达式。

专用合成检查可运行：

```powershell
& .\.venv-gpu\Scripts\python.exe -m pytest tests/test_quartic_memory_runtime_20261020.py -q
```

该文件的13个CPU/CUDA用例已在实际RTX4060Ti、Torch2.13.0+cu130环境通过；完整工程QA共30项通过（runtime13、研究runner8、native合同9），覆盖导出、非空Adam恢复、投影及梯度等价。QA来自固定本地源码31f100d39402ae7d78fd8c32c9525dc974d596f2；与main相比，六项依赖只存在CRLF/LF行尾差异、源码内容一致。30项包含未在本次PR合入的研究runner/native检查，不表示在main重新执行过完整QA。测试使用合成输入，CPU环境会执行对应CPU用例。

本模块为可选计算工具，部署默认模型保持原选择。候选结构的预测收益和采用条件见[研究协议](https://github.com/LLzijian/grid-transient-rom/blob/b85307f8ee41a4ee108ed515eb141e534e5e251a/reports/quartic_memory_20261020_protocol.md)。非负新增储能和局部谱检查不构成非线性稳定域证书。
