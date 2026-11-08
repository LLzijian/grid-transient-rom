# 严格JSON身份保存恢复：工程结论

新增适配器通过唯一一次合成CUDA检查。嵌套tuple转换为有序list后，保存与恢复使用同一独立快照；来源、代码、数据、配置的标量类型、参数顺序及bool/int变化仍被拒绝。原保存恢复工具和已关闭费用核验入口均未修改。

三个合成事件、float64、cuda:0，填充16个参数的Adam状态后，检查参数身份、参数值、Adam状态、CPU/CUDA随机状态、NumPy采样器和覆盖状态；恢复后的下一更新及梯度None模式通过原标准。浮点比较atol=rtol=2e-9，Adam step、整数和随机状态精确比较。无真实train、val或test访问。

复现命令：`.venv-gpu/Scripts/python.exe tests/test_closure_checkpoint_identity_20261019.py --output private_data/closure_mechanism_20261019/reserve/checkpoint_identity01/checks.json`。本轮已经执行，原输出目录禁止覆盖；复现应使用新目录并另行记账。预执行归档ecdc1731cee66c2043b71c1a2416375e2c2ba492，适配器SHA256为7761a88586bdf4e4e326071654693832339559836737c06ab5189e86082d23ac，测试SHA256为d4815ae0dd1988e346ffc1141f7ba9340147038df87f8aa3d57429e8fa2c3f14。

独立静态审查由/root/edge_result_challenge完成，未参与实现、未执行检查。执行前修正包含设备索引一致性，原helper的环境校验保持。root核对唯一执行回执及源哈希后接受新增工程接口，按范围限定PR集成；科学模型与默认入口保持原状态。

唯一检查2026-10-19 17:20:24至17:20:58（Asia/Shanghai），监督PID81348、启动PID86968，退出码0、未超时；启动及子进程合计一次34.5852151进程秒，stderr为空，已退出。原600秒分配关闭：实际34.5852151+支持保守保留300，未用265.4147849取消。原修复预留剩至少14065.4147849，父259200上限下剩至少231400.9211356。P0剩11753.4159736、P1剩3982.0903771及原阶段上限不变，旧900支持继续扣留。

[紧凑证据](../results/checkpoint_identity_20261019/summary.json)含原回执、结果与本地证据哈希；本地路径为private_data/closure_mechanism_20261019/reserve/checkpoint_identity01。合成临时模型不作为科学候选。此结果证明新调用方身份契约有效，不补齐旧费用门槛、不重开旧gate、不授予P2或模型采用。下一轮按[独立范围审查](research_scope_challenge_20261019.md)再实际调用Astra6，判断独立GRU构造/参数布局工程是否具有可执行、有限且独立的验收方案。
