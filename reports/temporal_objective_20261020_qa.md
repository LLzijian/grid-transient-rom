# 时间目标研究：工程检查

首轮native QA01实际32通过、2失败；费用33.8023423秒，inner/outer退出1、cleanupGONE、275项源前后一致。Torch2.13.0+cu130实际检测CUDA可用，设备RTX4060Ti，CPU与CUDA检查均实际运行。失败位于新测试调用旧one_update之前：旧linear_cache_cost_gate_20261019通过显式load_runtime初始化torch全局，新测试漏了该生产初始化。它不是新目标数值或物理失败。

协议唯一修复用于新测试setup补调用原load_runtime，旧源不改；测试断言AST逐项完全相同，没有改输入、数学定义、容差或采用门槛。完整原目标值/组件/全梯度/Adam、独立末段公式及其梯度检查均已通过，但整套QA仍须在新源身份下完整复验，不能以32项代替全套通过。QA01 stdout与回执身份归档；当前没有正式训练、prepare或cost。
