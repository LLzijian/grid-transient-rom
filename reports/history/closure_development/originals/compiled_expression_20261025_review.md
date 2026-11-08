# 表达式运行时：独立代码审查

**最终结论：PASS，保留为已通过所述CPU精确检查的显式启用工程候选。** 推理模式缓存问题已实际复现、修正并通过最终CPU回归；未发现当前两个generic edge候选的其他静态阻塞。本结论不提供GPU/完整训练资格、实测加速、main整合、默认集成或模型采用资格。

本报告由未实现补丁的代理撰写。只读三个新文件及必要的`symbolic_closure_rom.py`、`closure_training_runtime.py`等既有接口，没有运行测试、Python、GPU、作业、优化器更新、训练、预测或Git，没有递归委派；只写本文件。正在运行的training_efficiency01及其冻结来源不属于本补丁，不作修改或重新绑定。

## 已关闭的具体问题

原`ExpressionKernel._constant`及`_origins`以shape/stride/dtype/device或dtype/device作缓存key，在`torch.inference_mode()`下首次调用会缓存inference tensor；随后正常autograd以同形状复用时，乘法反传需要保存该常量，可能报“Inference tensors cannot be saved for backward”。旧解释器每次新建常量，不会发生这种跨调用模式复用。rollout结束清缓存的常见路径规避了此例，但直接kernel接口的错误已被root实际复现。

新增两项CPU回归的修正前会话8236退出1：节点kernel的推理→反传例失败，实际edge模型该例通过，另9项未选，pytest报告156.61秒。该失败按原source SHA及原文保留于`results/code_efficiency_20261025/inference_before_fix.json`，不能将它改写为两个例都失败或原版全通过。最终源码只将两类缓存常量的创建放入`torch.inference_mode(False)`；仍为普通无grad、无grad_fn常量，不改变表达式张量算术。新增断言直接核这三个属性并严格比较后续反传，已关闭该生命周期问题。

## 张量运算和回退

`ExpressionKernel`逐节点编译变量索引、const、add/mul、pow和tanh；保留原列表求值、逐项归约与乘法导数顺序，没有展开/合并多项式。端点反转、s变号、value/gradient平均及整体原点中心化与原`_torch_feature_eval`逐项对应。实际edge物理梯度中的cos切线修正、权重、投影及完整gradient拼接/切片顺序保留；不因当前系数为0跳过表达式。

新base只在没有额外storage、所有term均为generic node/edge时启用；空term、seeded快路径、port或混合配置整模型继承原实现，避免将不同算术路径混算。按需构造node/edge特征移除了当前两个edge候选不消费的node/port特征，没有删除被消费的特征或物理项。未发现当前支持条件下质量、角基准、q二次项或port回退的直接值错误。

## 缓存、复制与持久化

单位/零梯度及原点梯度只来自表达式元数据，不包含状态值或可训练参数计算图；导数运算使用新张量，port返回clone，避免外部直接污染缓存常量。独立kernel按最多24项常量缓存限制变化batch形状占用；模型rollout结束经`_clear_runtime`清除两类缓存。

kernel的自定义deepcopy重建闭包，避免复制Python函数后引用旧kernel/cache；`with_terms`重建kernel。继承的`_apply`、`load_state_dict`和`end_graph_cache`经动态调用新`_clear_runtime`清缓存，设备/dtype变换和状态加载不会继续使用旧常量。导出保留原symbolic格式，元数据缓存不进入state_dict；重新加载为新类仍构建新kernel。`adapt_base`拒绝活动runtime cache，复制原状态和requires_grad/provenance，保存并恢复CPU及相应CUDA RNG；它是显式新对象，未把旧runner改为新实现。

## CPU证据与主张范围

初版9项CPU通过由root实际运行并提供，非本审查执行。初版测试覆盖6项表达式的值/特征导数/反传、当前两个候选非零系数与非单位质量/非零角基准的field、RK4和8步小轨迹、field参数梯度、缓存复用/清空、kernel deepcopy、导出恢复及改项回退。初版未直接核整条轨迹的参数反传、推理→训练切换或设备/状态加载的清缓存断言。

最终会话54353退出0，11项通过，pytest报告29.60秒，optimizer/GPU更新均0；实际事实记录`results/code_efficiency_20261025/cpu_qa_final.json`的source/tests SHA与本审查最终文件相符。新测试补齐8步完整轨迹对可训练base参数的反传严格相等（原本未使用的参数均允许None）、推理→正常autograd、`to(cpu,float64)`及`load_state_dict`清两类缓存。最终通过、首次9项通过及上述不利回归均保留，未混为一次通过。本审查读取了这些事实与断言，没有自行重跑测试；不同会话的pytest秒数不是性能对照，也没有原生PID/Job成本可供补造。

这套测试读取本地horizon binding与冻结pool的语法元数据，未读取train/val数组；其通过依赖这些本地资产，不是仅检出main即可运行的自包含测试包。它没有验证4工况真实step2048、32恢复/Adam、完整50/200拟合轨迹、CUDA严格相等或GPU显存/速度门槛。当前field/短轨迹一致不能被扩大成完整训练已经等价。

代码检查报告正确区分带profiler开销、CPU局部等价与正式训练计时，也保留旧EdgePolynomial关闭分类；已有一个工况的profile只能支撑候选方向准备，不能推断完整加速倍数。新补丁不得自动注入仍运行的208文件冻结基准；后续GPU测量/采用须等当前作业关闭，并按另行明确的同源资格与工作量执行。

## 最终审查版本

| 文件 | SHA256 |
|---|---|
|`experiments/compiled_expression_runtime_20261025.py`|`fffa57f7ecee120684b8b4ca6d5f0e79142f9f1d1fc5f86171193f24a4ac9fa4`|
|`tests/test_compiled_expression_runtime_20261025.py`|`15cd33ea8c0d0d12bac71a0738c5805c1a73601ee665adc2dcfe8ee2e7abd8b1`|
|`reports/code_efficiency_inspection_20261025.md`|`44b794d6be001c99a18e291d1a1031f0db3f8666f13372933b34552816249d5a`|

最终检查报告准确区分初始、失败回归、最终CPU通过与未完成的GPU比较，明确本地元数据依赖和无实测加速主张。该PASS支持保留当前显式启用候选；待后续GPU、实际4工况/step2048、完整50/200轨迹、16事件、Adam/恢复与正式性能资格后再决定工程采用。它不自动授权接入现有作业、main整合、默认采用或科学模型晋升。
