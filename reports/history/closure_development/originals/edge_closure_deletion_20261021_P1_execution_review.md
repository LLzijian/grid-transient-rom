# 空项对照P1执行入口：独立静态检查

2026-10-21。只读新的P1 `edge_closure_deletion01/launch_stage.py`，不运行程序、测试、Torch、生成器或Git，不读取数据、权重或旧数值对象。依据root提供的实际P0 QA/cost已通过事实检查新增调度逻辑；没有重新认证数值QA。

**条件接受P1有限执行入口，未发现导致必然冻结失败或明确误报通过的新helper错误。** 最终新helper身份及P1来源须归档，root核实际配置与冻结源码后，才能生成并启动登记顺序。

- P0证明同时约束实际QA/cost原生成功、source前后相同、固定98条science及runner哈希、固定0a6ef来源、104条cost绑定表；admission的2450、数值来源、runner/tests哈希与实际证明一致，cost complete的字节/结果相同。新P1来源可以增加文档/helper，未放开科学代码哈希。
- old_selection的固定公开路径与d1706d146857946a771596adbc3576df5fb2b0014fd6f21ceb8a7aa9064a1c61在generate/execute均核语义键；三旧余弦端点的model/training六文件按原角色及哈希一并绑定，不只是保存一个未消费的files条目。
- 三train按5101/5102/5103独占目录、无重试；全部先前阶段实际通过才能继续，三有效才freeze，唯一freeze成功后才能按固定种子顺序做三次新评价。真正的三512/空项语义与train选择冻结仍由已QA冻结的科学入口执行，helper没有引入提前formal val路径。
- 500/480/450、freeze50/45/40、eval50/45/35与源码CLI一致。2450=1500train+150eval+100metadata+700support；metadata固定40+50+10，freeze只在metadata一次计费，没有再加到2450外。已发生阶段加剩余原生上限逐类别累计，不借分项。
- identity_bound在完整binding构成后、写binding及实际launch前均核：正式身份JSON字节+512×32+4096≤65536；读取实际QA fixture身份并核seed。generation先确定字节和哈希再纳入binding，没有binding自哈希循环；容量证明限train，不把freeze的seed=None错误当成训练角色。
- 成功条件要求两层实际COMPLETED、inner未超时、退出0、回执完整、源前后相同、整数FILETIME/PID/GONE；缺清理证据保留锁。phase_gate用于完成合同，评价的预测及数值资格仍须读取evaluation字段另行判断，不能把执行完成等同科学通过或模型采用。

此前root已在首次P1 generation前修补old_selection语义键及所有先前阶段通过要求；该静态修正不改P0数值冻结源码。没有发现需要重测或重开旧审计的理由。本记录不改变真实模型默认、项目val/test关闭、TASK4暂停或科学门槛。

工具：只读命令1.8795812秒；本次写记录回执另计，合计上限5秒，计P1支持700。所有接受意见均限本次读取的新helper；实际配置和最终文件哈希由root在执行前核定。
