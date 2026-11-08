# 检查点续接结果：独立只读审查

审查对象为源`6bf8f730e4fc180af9e815682a8fb4fe4b7fd020`的已完成执行。本次只读取协议、完成标记、冻结清单、评价结果、原生回执、已保存训练轨迹及相关源码；未运行测试、GPU、模型评价或Git，未反序列化`training.pt`。仅新增本报告。

结论：本轮完成了原12模型的512步比较和一次合成val，支持“已知有理形式在本轮固定共同训练机会下取得较低val归一化MSE”的研究结论。结果和来源报告可以作为科学进展进入main；不据此采用默认模型，不将已知形式比较解释为未知结构发现。新增恢复/资格模块暂留归档分支，当前不作为依赖自足的工程包合入。

## 完成与恢复证据

`rational_checkpoint02/run/complete.json`记录12端点冻结、6048次新增正式更新及132次工程验证更新。`science/selection.json`包含12个固定seed/form/noise端点，所引用的12份训练轨迹均长512；三份`0_rational`身份明确连接旧源`5ffe63f5f9bd71c68e417da4759fdbd183299799`的32步来源。历史96次正式更新加新增6048次为6144，132次验证未计入正式进度。

原生`native_run/receipt.json`为COMPLETED，三个退出码均0，Job最终0、无遗留清理，wall为2673.6449287秒，Job CPU user为1883.703125秒、kernel为589.5秒，二者与wall分别记账。实际来源前后快照均为**182项**且相同；不能将本次原生快照写成179项。完成标记与评价记录同源。

已保存的单位摘要为真实共享serializer的12项通过；CUDA资格摘要记录四路径各33更新、共132更新，32步发布/真实恢复和第33步计算状态逐值一致，旧6301真实32步状态对照一致，三份来源迁移均通过。上述证据支持修复与续接的实际成功，不改写原失败组及其检查点，也不把旧QA06改称新源通过。本次独立审查读取这些证据，未另行加载状态或重做预测。

## 科学比较与结论范围

指标为干净coarse未来上的**归一化MSE**，无物理单位，不是RMSE或训练损失。每模型train为16事件、未来0.5秒；合成val为16事件、未来1秒；采用原train尺度。所有12项已保存train/val数值资格均为true。以下为val全部16事件、全部30观测变量的均值，配对差定义为`tanh − rational`。

| 噪声 | seed | rational归一化MSE | tanh归一化MSE | 配对差 |
|---|---:|---:|---:|---:|
| 0% | 6301 | 0.0005136093 | 0.0005764934 | 0.0000628841 |
| 0% | 6302 | 0.0010422507 | 0.0011023605 | 0.0000601097 |
| 0% | 6303 | 0.0007516918 | 0.0010117306 | 0.0002600388 |
| 1% | 6301 | 0.0019319434 | 0.0019446316 | 0.0000126883 |
| 1% | 6302 | 0.0025004254 | 0.0025624565 | 0.0000620310 |
| 1% | 6303 | 0.0018472362 | 0.0021151812 | 0.0002679450 |

六项方向均为positive；0%噪声三个实例的平均配对差为0.0001276775324，1%为0.0001142214343。这是六个预注册条件中的共同方向，不是六次独立优化重复：不同seed同时改变数据与初始化；两个噪声条件共用各seed的干净事件来源。每一对相同seed/noise的两形式，其train观测摘要和val原始数据摘要一致，配对成立。不能把96个val事件当作96次独立训练重复，也不能从本轮给出跨数据分布的显著性或普遍领先结论。

共同512更新不保证两形式同等收敛。读取已保存loss轨迹发现，12模型各自第481–512步的训练损失均值都低于第449–480步均值。这是随机batch下训练仍有改善的观察，不是新的收敛检验；按原协议应保留“固定更新预算下尚未建立收敛结论”的解释。不能把配对差当作tanh表示能力下界，或把两形式差异完全归因于单个物理机制。

该研究预先提供正确函数形式，未向模型提供真系数、隐藏状态或RHS。结果支持已知形式先验的预测价值；不证明未知函数发现、真实系数识别、能量/稳定性结论，亦不替代原路线中尚需独立证据的目标。模型继续为研究候选，原项目val/test和TASK4限制不变。

## main集成范围

已读取保存的`windows_runtime_main_verification.json`：PR30记录main合并提交`3cab35e4bb9bfe29f282d3ea16267f574d7dabef`，其六文件范围只包含Job/helper、16项运行环境测试及相应报告/摘要，没有旧完整训练入口。本次未调用Git或远端接口再次验证main。

当前新增恢复模块第14行仍导入旧`rational_execution_20261021`的`atomic/new_or_equal`；qualification与新增测试继而依赖该模块。因此不能直接把两个新增模块和测试三文件称为main上的自足工程包，也不能为了满足该导入顺带默认合入已知诊断命名故障的旧完整core。共享数学/serializer依赖属于此前已验证闭包，不需要为本轮重复修改。若未来单独集成模块，须先精确解决这个小型工具依赖并验证最终差异；本次不新增该修复或验收。

按root最新决定，本次可采用以下**结果/报告六文件**候选范围，待root核对最终形成的正文与事实文件后提交：

1. `reports/checkpoint_continuation_20261022_conclusion.md`
2. `reports/checkpoint_continuation_20261022_protocol.md`
3. `reports/checkpoint_continuation_20261022_result_review.md`
4. `results/rational_execution_20261021/checkpoint_completion_facts.json`
5. `results/rational_execution_20261021/checkpoint_unit_summary.json`
6. `results/rational_execution_20261021/checkpoint_cuda_summary.json`

科学端点、完整评价、原生回执及迁移来源通过固定归档提交/摘要链接追溯，大文件和旧负结果继续保存。结论与事实文件须准确写入新源、旧96/新增6048/验证132、12端点、一次合成val及182项原生来源；不得把它们写成新模型默认采用。私有launcher、binding、旧资格原件及本机环境资产不在此六文件范围，也不能宣称这些报告构成可在干净checkout直接运行的科学入口。

**采用判断：科学结果与报告范围可集成；当前模块三文件作为自足工程包仍有上述明确依赖阻塞。** 本判断不授权新增执行，不修改既有协议或运行源。最终六文件差异、归档链接、expected head及远端合并由root核实；本报告不声称已完成这些Git操作。

## 精确main六文件终审

**PASS：同意以下精确六文件作为科学结果、协议与复现证据范围进入main。** 本清单取代前文候选清单；压缩完成摘要替代候选中的本地完整facts，完整facts仍按原摘要保留本地。

1. `reports/checkpoint_continuation_20261022_conclusion.md`
2. `reports/checkpoint_continuation_20261022_protocol.md`
3. `reports/checkpoint_continuation_20261022_result_review.md`
4. `results/rational_execution_20261021/checkpoint_completion_summary.json`
5. `results/rational_execution_20261021/checkpoint_unit_summary.json`
6. `results/rational_execution_20261021/checkpoint_cuda_summary.json`

实际读取的完成摘要为54563字节、12模型、COMPLETED、issues为空。12项train/val全部变量平均归一化MSE与原`evaluate01/result.json`逐项相同；六个原配对差、原评价来源及完整facts的大小/SHA引用保持可追溯。绝对`reference_base`与各引用的`relative_to`明确说明本地资产解析方式，并未把这些数组、training.pt或私有入口声称为六文件内自带内容。本次未重新核全部145项资产摘要，也未反序列化状态或重做预测。

单位与CUDA摘要均为实际源`6bf8f730e4fc180af9e815682a8fb4fe4b7fd020`的PASS：12项CPU检查、132次CUDA资格更新、三份迁移及全部登记状态比较一致。结论中的14.24%和5.17%明确为每噪声条件两组三实例均值的相对降低；正文保留不同seed数据/初始化共同变化、已知形式而非未知发现、共同512更新不等于同等收敛、无统计显著性或模型默认采用的范围限定。182项原生来源及wall/CPU分别报告，未混淆历史96、正式新增6048与工程132次更新。

**本次不集成依赖旧core的三个新工程文件：** `experiments/rational_checkpoint_recovery_20261022.py`、`experiments/rational_checkpoint_qualification_20261022.py`、`tests/test_rational_checkpoint_recovery_20261022.py`。亦不集成旧已知命名冲突的`rational_execution_20261021.py`或私有launcher。main获得本次可核对的研究结果，不获得一个宣称自包含或默认启用的训练/迁移入口。

允许root将结论中“事实摘要”链接指向本清单的`checkpoint_completion_summary.json`，保持科学语义不变。最终实际PR六文件差异、固定归档来源、expected head及远端合并由root核实。本次终审仅补写本节，未修改其余五文件或任何运行源码，未运行测试、训练、torch.load或Git。
