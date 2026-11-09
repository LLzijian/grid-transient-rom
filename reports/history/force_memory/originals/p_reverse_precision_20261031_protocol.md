# p_reverse_precision_20261031：已失败反推门的数值来源与影响

状态：DESIGN_FROZEN；root已采用Scientist有界设计，执行待实现与独立审查。
2026-10-31；Campaign force_memory_closure_20261030；目的probe。
父Run p_port_diagnostic_20261031已提交并普通推送，远端核验提交
f48987abcc10f5361194ca96419a197f62fa0269；私有archival_verified.json保留核验。
用户恢复既定方向内自动推进的授权与数据/部署边界沿父Run，不逐轮重新询问。

## 固定设计与root选择

规范设计：[p_reverse_precision_20261031_scientist.md](p_reverse_precision_20261031_scientist.md)，
SHA256 b3be4d5dca56e13123a05279975d6b08b808c158bb664700525729d7512b5d93。
其§2–§6的输入、固定组合、高精度资格、传播核对、解释和停止条件全部作为本协议规范内容。
旧反推buffer的1e−12门已失败，本Run不改旧门、不追认旧M3、不改变原模型/数据。
root选择本probe是为了判清失败门的来源与初始化影响，避免把数值差异直接当物理闭合问题。

## 有限工作与合法信息

固定A96与A16的模型常数与原缓存中的11点因果δ/ω和末点真p₀；
只读NPZ的event_ids/histories成员，不读取fine/coarse或未来真值。
初始化历史前10点p置零占位，不使用历史p真值。fit297全部事件保留，不选子集。
原val/test关闭、新FOM=0、TASK4暂停、default none和所有模型指针不变。

P0核实际加载类/调用行与p_backward消费；P1固定binary64 Q、80/120位参照及独立3×3解析，
固定Torch64/SciPy64/Torch32指纹组合与JSON/装配舍入核查，不扫描算法或参数。
P2最多2模型×2buffer×2历史步长=8组全队列初始化及对应初始场读出；
只在丢弃的内存副本替换p_backward，原对象和其余state_dict不变，不写候选、不导出模型。
核p历史线性传播、QR对r₀的传播及初始导数差；不把初值误差外推成长期预测误差。
optimizer=0、共享参数估计=0、新ROM rollout=0。历史因果求解不算新参数拟合。

高精度参照不合格或实际路径/身份不明时停止依赖项；纯实现错误最多一次有界修复。
旧门未通过不构成修复许可，不放宽阈值或重置旧余额；未知作业先对账不重复启动。
执行包装日志独立放private_data/p_reverse_precision_20261031_execution，
不得在工具启动前创建其不可覆盖的private_data/p_reverse_precision_20261031结果目录。

输入/设计/源码哈希、真实执行身份与退出回执先登记后执行；原生身份只留私有。
完整结果留私有Run目录，公开聚合与哈希写results/p_reverse_precision_20261031/。
结束分别作代码、模型、科学决定并独立审查、选择性提交和核验远端。
任何采用解析buffer或资格修订须另有前瞻协议与原数值/关键事件核查，不由本probe自动授权。
