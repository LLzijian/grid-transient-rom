# 删除新增边储能：独立实现静态审查

2026-10-21。只读新入口、新测试、P0两个小监督文件及原with_terms/from_config的必要函数片段，没有运行Torch、程序、测试、生成器、推理、Git或读取数据/权重/轨迹。审查者未参与本次提案和实现。

**结论：修正唯一cost的保存计时路径后，可进入已登记的唯一QA。** 这不是数值QA通过或模型采用结论。

## 必须修正的计费合同

新cost手写`m.export + torch.save(cost_only_training.pt) + cost_only_save.json`，保存identity只有cost_only/actual_updates；正式训练调用原`checkpoint()`，携带完整binding/source/seed/fixture，并经过pending目录、原子发布及latest/hash写入。cost附加8192字节trace padding没有证明覆盖被省略的完整binding身份和实际发布操作。因此当前17×save估价尚不能称为同格式正式保存路径实测。

建议在cost单独目录直接调用原checkpoint，使用真实step3、三条真实有效trace，identity含与正式等量的binding/source/seed/fixture，以及cost_only/actual_updates=3和明确额外padding来保守覆盖512条trace的开销。checkpoint自身允许step3；32倍数限制属于旧restore，cost仍用现有step1专用snapshot恢复，不修改历史函数。不得把cost声明为512步或将其作为科学起点。如此即可实测原子发布、哈希和完整身份写盘，避免扩展成新保存平台或额外一次费用试验。原科学512及17保存政策不变。

## 其余已核实的新路径

- `runtime_model(source, [], seed)`明确给空集合，assert_empty检查terms为空且coefficients无元素；with_terms确实按传入列表设置项。新测试核空项初始与训练后导出回读，防默认填回旧项。
- QA合同逐种子比较旧共同stage0，合法排除唯一coefficients形状差异，核其余state、参数顺序/flags、空Adam组配置、NumPy/CPU/CUDA RNG和六头零值；初始场和共同原始梯度比较不强制跨结构裁剪/Adam相同。
- 新空项generic参考复用已验证AST值/autograd完整场实现；tests有非单位质量、0.5秒损失、所有原始梯度、裁剪、Adam、非空恢复后下一更新精确一致、导出及一次11点初始化/未来拒绝。完整轨迹路径自带缓存建立/清理；不需要重开未改旧实现的全库测试。上述是静态覆盖，不代替唯一CUDA QA。
- QA/cost/fit只调用已限定stream0的train生成，正式stream1只在评价分支。cost三种子各4次更新计算，恰12；逐种子最慢值、三个里程碑、25%余量、450/35及2450完整帽在代码中明确。
- 正式三个空项512、17保存、完整train评分合同保持。barrier在train选择前校验三个完整空项端点；评价再核完整冻结清单及模型语义后才生成正式val。旧两式train候选来自明确old_selection绑定；P1生成器必须按root已指定路径和d1706d146857946a771596adbc3576df5fb2b0014fd6f21ceb8a7aa9064a1c61固定，不能选择其他旧表或改旧结果。
- P0小监督保留唯一QA→cost、相同immutable来源、旧固定allowlist、前后hash、整数FILETIME/GONE及缺清理证据留锁；210/230/250和1200分项匹配。P1仍须P0实过后另登记，不由入口存在自动授权。

root须在QA前归档最终静态修正和来源身份，随后遵守唯一QA/cost及失败不修源重试规则。没有发现需改变科学问题、状态、损失、物理单位、事件/数据权限或旧历史门槛的问题。新结果仍只是开发合成删除对照，不晋升真实模型。

工具前三次只读命令原生时长1.8018885、1.9177507、2.0141703秒，共5.7338095秒；本次写审查文件回执另计入新P0支持500，累计上限10秒。第一次输出截断，第二次补读完整测试及make头部，第三次只补入口尾部和原空项构造必要定义；没有执行被审查源码。

## root采纳的QA前有限修正

root已接受上述具体问题，拟在新cost调用原checkpoint：真实step3、真实step1/恢复step2/step3三条trace，完整source/binding/seed/fixture及EMPTY身份，加cost_only/actual_updates=3和保守64KiB padding；实际目录checkpoint_00003，不伪报512。P1生成器启动前另以静态JSON/序列化大小证明实际正式身份与512trace开销不超过cost载荷及64KiB余量，正式保存不加padding，不新增Torch试验。该修正若在唯一QA前落实、静态身份/容量证明通过并重新归档，便解决本审查唯一阻断，接受进入唯一QA。此处记录root修正承诺，不自称已重新读取修改后的源码或已核容量证明；最终核验由root完成。
