# GPU训练与运行效率

后续40维闭合模型训练使用`experiments/train_closure.py`。默认设备为CUDA，CPU模式需明确指定`--device cpu`。历史实验入口及已冻结结果保持原样，服务原方案的复现；新的训练使用新输出目录。

## 环境

本机GPU为RTX 4060 Ti，显存8 GB；CUDA环境为项目内`.venv-gpu`，原研究环境保持独立。Windows安装入口使用已有Python创建独立环境，并从PyTorch官方源安装`torch==2.13.0+cu130`：

```powershell
./scripts/setup_gpu.ps1 -Python 'E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe'
& '.venv-gpu/Scripts/python.exe' experiments/train_closure.py doctor
```

安装方式依据[PyTorch官方说明](https://pytorch.org/get-started/locally/)。CUDA不可用时入口会报错，不会静默转回CPU。环境及下载文件不进入GitHub。

## 训练与恢复

训练命令的必填参数是`--source`、`--output`、`--updates`、`--max-seconds`；分别指定已登记的源模型JSON、新输出目录、累计更新次数和执行时间上限。`--seed`默认为2701；`--final`启用登记方案的512次之后学习率衰减。源格式为`symbolic_closure_v1`，包括100 ms初始化器。该命令只使用既有train读取器，不加载val或test。

```powershell
& '.venv-gpu/Scripts/python.exe' experiments/train_closure.py train --help
```

恢复时使用相同参数与输出目录，加`--resume`；`--updates`表示累计目标，不表示额外追加次数。只有模型、train队列、代码、学习率、精度和来源身份吻合才恢复。更换CPU/GPU、GPU型号或PyTorch环境需要明确加`--allow-device-change`，并记录为数值续训，不声称逐位重放。相同环境恢复覆盖模型、Adam动量、事件覆盖顺序、NumPy随机状态以及CPU/CUDA随机状态。

每个保存点是独立目录`stepNNNNNN`，`latest.json`在该目录完整写入并核实哈希后更新。保存先写唯一临时目录，再发布完整目录及指针；中断留下的临时目录和未被指针引用的目录保留，不阻塞重试。异常状态单独保存在`failed_state.pt`及`failure.json`，不会代替上一个有效保存点。参数更新前的中断恢复本批采样状态并保存最近完成步；Adam更新中的中断回退到最近盘上保存点，并明确记录丢弃的已完成更新数，不声称逐步无损。时间上限到达时保留已完成的更新。旧研究的冻结目录不能用作新输出目录，也不把旧72进程小时余额自动转作新实验。

## 执行优化

- 真实变长事件一次驻留CPU或GPU；训练批目标和行索引在积分前准备，积分循环内不再逐步做NumPy堆叠或CPU到GPU复制。
- 相同有效长度的区间分段推进；失稳样本只使用真实观测范围，不补尾，不删事件。
- 同一次场计算共享角差与端口特征；矩阵与参数相关常量每个反向图构造一次，更新参数前清空。一般合法表达式保留原解析编译路径。
- 同一局部历史的初始化结果在该训练目标中复用；自主预测仍只初始化一次。
- 学习率读取、参数列表和设备数据准备移出积分热循环；有限值检查合并GPU标量同步。保留非有限值、梯度裁剪和失败保存。

所有运行保持float64、0.01 s RK4步长、原1/2/5 s训练安排、原事件采样与轨迹风险定义，不默认启用低精度或改变批大小。总目标明确等价于`discover_closure.loss`：10×轨迹风险+局部轨迹+0.1×隐藏一致性+0.1×转矩辅助，不包含0098原始objective中的系数L1项；这延续已登记符号发现目标。CPU和GPU的逐位结果可能不同，正确性由场、损失、梯度和恢复测试检查。

## 性能与验证

可复现的合成基准命令：

```powershell
& '.venv-gpu/Scripts/python.exe' experiments/train_closure.py benchmark --batch 12 --horizon .2 --repeats 3 --output private_data/gpu_benchmark.json
& '.venv-gpu/Scripts/python.exe' -m pytest -q tests/test_closure_training_runtime.py tests/test_train_closure.py
```

基准同一组数据与参数，对比原CPU、优化CPU及优化GPU，并保留“目标已驻留GPU、场仍为原实现”的同设备对照。计时包括每批准备与前向反向、排除一次性驻留、保存和优化器更新；CUDA计时显式同步，首次预热不进入中位数。96事件的附加测量考察吞吐量，不改变原训练的12事件采样。精度与性能结果见[本轮工程结论](../reports/gpu_training_20261017.md)。

## 登记八／七项结构的参数缓存

`experiments/linear_cache_runtime_20261019.py`提供显式的`enable_linear_cache(model)`，在原模型对象上启用每个训练图的参数乘积缓存，保留Parameter身份、优化器引用和原导出格式。只接受登记MAIN8及删除完整边余弦项后的七项、40维L10-S，无额外储能表达式；其他结构会拒绝。现有训练入口保持，新的已登记研究可显式使用该接口。

训练继续用原`graph(model)`上下文，反向后退出上下文并清缓存，再执行Adam；每次更新重建参数派生量。无图评价使用当前参数即时计算，导出前必须清缓存。通过原helper恢复后，可在同一模型对象上启用；不要替换优化器持有的Parameter。

本机float64、固定12事件、1/2/5秒完整更新负载的主计时指标分别缩短16.69%和17.56%。两结构的完整目标、全梯度、连续更新及保存恢复检查通过，详见[前瞻协议](../reports/astra6_linear_cache_20261019.md)与[限定工程结论](../reports/linear_cache_20261019_conclusion.md)。这是同GPU运行路径比较，包含Adam，口径不同于上节0.2秒基准；真实研究费用须另行核算，工程采用不改变模型或test门槛。
