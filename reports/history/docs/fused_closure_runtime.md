# 固定闭合方程的公共乘积合并

`experiments/fused_closure_runtime_20261021.py`为现有N七项和D九项提供可选运行实现。它在一次完整优化图开始时，将储能梯度、交换/耗散与端口的参数公共乘积预乘到输出映射；每个RK4子步复用这些仍连接自动微分的矩阵。方程、参数、训练掩码、float64、RK4步长、目标和优化器保持原设置。

```python
from fused_closure_runtime_20261021 import enable_fused_runtime, one_update

# model必须是现有40状态N7或D9模型，且没有活动中的优化图。
enable_fused_runtime(model)  # 保留原参数对象及优化器引用
result = one_update(model, optimizer, store, samples, horizon, force_scale)
```

运行类只接受登记的N/D表达式身份；D的四次储能系数a保持非负，三次动量端口系数b保留符号。外部u直接替换完整自主端口，储能贡献继续生效。缓存仅属于一次优化图，反向后清理；加载、设备迁移和异常时清理；导出要求优化图已关闭。无图调用重新计算映射，不保存过期参数值。导出格式保持原样，原运行类可以读取。

固定本地源码`6abbc9d853fdcab3799fda7037184ab2570ef78d`已实际通过42项QA：runtime33项、合成审计与原生监督9项；CPU/CUDA各12个参数化用例均执行。检查覆盖完整5秒原目标和全部参数梯度、同参数裁剪/Adam/仅a投影、非空Adam与CPU/CUDA/NumPy RNG文件恢复、非单位惯性、完整图和骨架重编号、外部输入、导出、缓存及真实可用长度。两层退出0、79项来源前后一致、进程清理核验通过。

唯一GPU计时使用冻结的12条531点合成序列，N/D两结构、1/2/5秒三个时域，AB/BA两个区块，每条路径首次加两次连续重复，共72次完整更新。每个时域合并两个区块的4次重复取中位数，首次耗时另行保留。环境为RTX4060Ti、Torch2.13.0+cu130、float64；计时包含缓存构建、原目标、反向、clip100、Adam及必要投影。

| 结构 | 时域 | 原运行类中位数/秒 | 合并后中位数/秒 |
|---|---:|---:|---:|
| N7 | 1秒 | 0.9374693 | 0.5329576 |
| N7 | 2秒 | 1.7104300 | 0.8948287 |
| N7 | 5秒 | 4.0148565 | 2.1490632 |
| D9 | 1秒 | 1.1689939 | 0.5859279 |
| D9 | 2秒 | 1.9670193 | 1.0478745 |
| D9 | 5秒 | 4.4989682 | 2.3965677 |

三时域中位数之和的用时分别下降46.3158%和47.2118%；两个结构均超过登记的10%要求，六个时域均通过不增加超过5%的条件。36组配对记录的损失最大绝对差1.1102×10⁻¹⁶，梯度范数最大绝对差3.3307×10⁻¹⁶。全部首次、重复和构造/保存耗时保留于原JSON，未删除慢值。这是上述固定合成完整负载的实测收益；真实训练仍须独立完整成本核算。

本地runtime检查入口如下，必须提供已冻结的合成夹具目录：

```powershell
$env:FUSED_CLOSURE_FIXTURE = 'E:/03-nn_Transfer_Func/kiki/private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01'
& .\.venv-gpu\Scripts\python.exe -m pytest tests/test_fused_closure_runtime_20261021.py -q
```

runtime测试通过新audit读取`manifest.json`、`model_N.json`、`model_D.json`、`synthetic.npz`及`rng.pt`，不读取真实轨迹或科学模型权重。夹具规范、生成代码、来源及每个文件SHA见归档分支`codex/fused-closure-runtime-20261021`和`codex/fused-closure-runs-20261021`下的`results/fused_closure_runtime_20261021/prepare01_{manifest,result}.json`。数组、RNG和重复试算checkpoint保留本地；恢复完整受监督命令还需要归档`reproduction`中的三个私有辅助入口按原BASE路径放置。原命令/绑定含固定工作区身份，异机复现应重新核对路径和来源，不覆盖既有证据。

完整42项包含未随此工程交付默认运行的审计/native设置，不表示main上的自足完整CI；本地实测环境和来源身份如上。该模块保持可选，默认模型不变。旧D成本拒绝、各模型的预测结论与科学采用门槛保持原记录。
