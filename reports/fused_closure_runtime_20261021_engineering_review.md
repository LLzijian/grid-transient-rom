# 融合闭合计算：独立工程采纳审查

本代理未参与提案或实现。本次只读新runtime、audit、runtime测试、说明、实际QA/基准JSON和stdout，并通过GitHub只读API核固定main依赖。未运行测试、Torch、模型、资产生成或额外基准；未读权重、数组或科学数据，未修改任何绑定文件、旧报告或协议，未递归委派。仅新增本文；root保留PR和科学决定。

**支持精确四文件的范围受限工程PR，未发现依赖或正确性阻断。** 文件为`experiments/fused_closure_runtime_20261021.py`、`experiments/fused_closure_audit_20261021.py`、`tests/test_fused_closure_runtime_20261021.py`及`docs/fused_closure_runtime.md`。说明有一处缓存措辞应按实际接口细化，见后文；不需要源码修改或新增QA。

## 固定main与依赖

GitHub原始commit响应确认基准main为`e815a0eb736c8cca79ac427cca1bccffba8d9410`，第一父为`8bc36d93f2885985fa4a0f30cca847f16a4b886c`，此次差异仅新增D runtime、其测试、说明三文件。此前独立审查的38个旧依赖见证继续适用：32项字节相同，六项仅CRLF/LF；本次不将该历史见证冒称重新扫描全仓。

针对本次新增audit的依赖及直接计算入口，本次另从固定main逐个读取九份已解码源码，与local逐字符完全相同，无需行尾规范化。独立计算远端UTF-8内容的Git blob SHA1，全部与API返回身份相同：

| 依赖 | 固定main blob SHA |
|---|---|
| `experiments/linear_cache_cost_gate_20261019.py` | `dd35a76cae236025a01bc0efb490f76512f96240` |
| `experiments/linear_cache_runtime_20261019.py` | `c6e373aed4a85eefd0014b2d3395b588b760754d` |
| `experiments/closure_p3_disagreement_20261020.py` | `995edf4dd840903ea034dfa3f7b94ade86d2898e` |
| `experiments/cubic_memory_port_runtime_20261021.py` | `ec6fb72ccbb3f7712a117282a2f2d65b8df863de` |
| `tests/test_quartic_memory_runtime_20261020.py` | `9559e6d61d3590ffad349188c9014106ba44ee50` |
| `experiments/closure_training_runtime.py` | `2ea984a3058b9685cb5a327662f1d7697bc20426` |
| `experiments/symbolic_closure_rom.py` | `e3a1bfff8dda6667f664d208e503c52ba82034d4` |
| `experiments/train_local_memory_0098.py` | `c2d6ddc127753a05212ec59a2a62a1d5353ed3a8` |
| `experiments/train_closure.py` | `559387e1045b64c7d13fa8f816f756090ce5ee78` |

audit的新入口只从旧N cost helper使用`load_runtime/one_update`，其计算再调用已核原目标和graph；没有调用其科学来源、cost或数据入口。Windows身份helper的顶层为标准库，本路径只调用FILETIME函数，不运行其科学freeze/infer。准备使用main已存在的quartic合成source及toy_source；其原有计算依赖属于此前明确见证。不存在为四文件PR顺带带入其他旧源码改动的必要。

## 实际QA与完整更新证据

固定源码`6abbc9d853fdcab3799fda7037184ab2570ef78d`的唯一QA01实际42通过/0失败：runtime33项、audit/native9项；CPU和CUDA各12个参数化用例实际运行，环境RTX4060Ti、Torch2.13.0+cu130。`qa01_result.json`与原stdout一致；原生44.01498919998994秒，两层退出0，79项绑定前后一致，清理GONE，没有QA02。先前静态发现的旧N初始化缺口已在新audit中显式调用`load_runtime()`，新增QA实际调用冻结N资产的旧更新入口，未修改历史helper。

QA支持原场、外部u、状态和全部参数梯度、完整5秒目标各项、裁剪/Adam/仅a投影、非空Adam及CPU/CUDA/NumPy RNG文件恢复、非单位惯性、完整图与骨架重编号、缓存和因果初始化的工程正确性。融合实现保留方程和参数身份，公共乘积仍连接自动微分；外部u替换自治端口并保留储能。没有依赖零a/b跳支、detach或低精度改变工作量。

唯一基准的公开原始JSON SHA256为`5bb807c5544875d9a786a4e295cd4de738f7d370a21b4ea766e9577c9358a999`。本次只读核对实际72条记录：N/D×1/2/5秒×AB/BA×old/new的24组，每组均有rep0/1/2。构造和保存各24次，首次记录仍在，四次重复中位数口径未变。原生140.20912730001146秒，两层退出0，85项来源前后一致，worker及supervisor清理GONE，记录与QA共享同一fixture manifest身份。

| 结构 | 三时域中位数之和：new/old | 用时下降 |
|---|---:|---:|
| N7 | 0.5368423603348653 | 46.315763966513465% |
| D9 | 0.5278821111511006 | 47.21178888488994% |

六个独立时域比值为0.5012240868至0.5685066701，均≤1.05；两结构总比均≤0.90，登记速度门槛通过。`root_numeric_verification.json`与基准提取结果记录36个配对、零数值不符，损失最大绝对差1.1102230246251565e−16、梯度范数差3.3306690738754696e−16；没有更改容差。本次未重新执行提取器或加载试算checkpoint。基准JSON没有N的完整Adam状态，完整参数/Adam恢复的依据来自实际QA，不能将36个摘要比较改称额外检查了全部checkpoint。

该结果支持固定12条531点合成数据、float64、相同目标/完整更新、固定GPU环境下的速度收益。它不直接给出真实研究训练的全预算，不重开已关闭D成本轮，也不证明预测改善、机制发现、模型采用或非线性稳定。

## 交付范围与说明

audit是runtime测试读取冻结资产的直接依赖，因此四文件比先前三文件假设自洽。runtime测试仍要求实际`prepare01`合成资产；完整42项中的audit/native测试、三个私有监督helper及数组/RNG文件留在归档与本地，不属于main自足CI。说明已列出这一区别、归档分支、资产规范/哈希、恢复私有入口和异机路径核对要求，没有声称仅clone这四文件即可运行完整42项。

建议将说明中的“加载、设备迁移、异常和导出也清理”精确改为“加载、设备迁移和异常时清理；导出要求优化图已关闭”。实际`export`在活动图上拒绝调用，不能解释为自动关闭调用者的图。其余已核文案与证据范围一致；root可只改未绑定说明，不改已执行源码或测试。

当前runtime SHA256为`8f332f72fb979413dd9dd2a7cd045b89bcc387253fd7915e6e2f985df6069940`，audit为`4abd8aeee3bf396db2bf7d12bcf07d3783e3dd24c53e84e2dd7d0201293e16c0`，runtime测试为`c592e896817478b69f628c137847d3413074637d051e8d5ffd5b2202576bab19`，均与实际QA/基准版本相符。所读说明为`06e3362794e4379fb873bb1b22e39717c5b837ac47ba15058e6164ba2cb8e953`，上述小文案修正后由root登记最终blob。

合并前由root核四文件精确blob、main expected head、依赖未发生新变化和远端结果；不整体合并归档分支，不带入私有helper、资产、其他历史修改或默认模型。实际本地QA加固定main源码等价是本次接受依据，不称已经在main checkout重新运行QA。

本次工程审查只读子进程实际16.5916659秒，包括一次结果转发表达式报错但已完成的2.0099387秒文件读取；该费用没有遗漏。加本阶段先前审查11.5623202秒为28.1539861秒，全部为既登记P0支持600秒的子集，不重复新增分配。GitHub只读调用及本报告编写仍属于同一支持hold，无新增原生/科学作业。本文写毕停止编辑。
