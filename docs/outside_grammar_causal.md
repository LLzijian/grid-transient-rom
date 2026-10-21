# 语法外真值的因果预测试验工具

该工具建立固定40维网络上的有理端口真值，并将前30维观测交给合法tanh端口模型；模型用100ms历史初始化一次，随后自主推演。它为“候选语法之外的真值能否被有效近似”提供可复用的受控试验，学习器没有隐藏q、RHS或干净导数标签。当前没有正式预测或结构发现结果。

9项CUDA/CPU检查通过，覆盖逐RK4子步的真实端口、物理作用方向、完整0.5秒梯度与实际clip/Adam、NumPy导出、因果初始化、完整16事件、真实32→34恢复及三seed冻结前禁止生成val。冻结检查读取实际训练计数、trace、来源、非空Adam和文件SHA；目录名写512不能冒充真实512步。

唯一完整费用测量保留96个更新（包括首次开销）、12保存、3恢复、6构造、3加载、三组train/val生成与完整双步长评价，以及全部写盘/来源启动费用。按预登记最慢值计价，三seed完整研究预计1957.8671进程秒（含25%余量和180支持），超过900秒上限；因此正式训练与合成val均未启动。这个结论是当前运行条件的费用准入结果，不是模型的预测或表达能力结论。旧数值标准、原项目val/test关闭及TASK4暂停保持，默认模型不变。

[专用入口](../experiments/outside_grammar_causal_20261021.py)、[检查](../tests/test_outside_grammar_causal_20261021.py)和[紧凑证据](../results/outside_grammar_causal_20261021/capability_profile.json)对应冻结源e761605f2d328dc3be97458863c5df0671eceea9。[完整设计、原生回执、全部耗时和准入计算](https://github.com/LLzijian/grid-transient-rom/tree/cb04439bf744e6560d9a6c5f6242cd5d4276be08/results/outside_grammar_causal_20261021)位于归档分支，source protocol和私有Windows监督助手须按原路径/来源复原。Torch2.13.0+cu130、Python3.12.14、RTX4060Ti、float64为实际检查环境；CPU参考属于工程检查。原始数组和真实Adam checkpoint留本地，路径/字节/SHA已归档，不能仅凭本说明启动未准入训练。未来研究需新的有限协议与费用准入。
