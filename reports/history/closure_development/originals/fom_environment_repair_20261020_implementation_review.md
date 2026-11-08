# FOM环境接线：执行前有限实现审查

2026-10-20，独立代理 `/root/fixed_edge_final_review`。只读新环境模块、九项guard合成测试、本地QA启动器和其直接attestation接口；未运行、导入、编译、求解，未读取权重、事件轨迹或旧val/test结果。仅新增本文，不修改生产源码。

主体实现符合本轮无求解范围：真实FOM身份核对固定解释器/prefix、NumPy/ANDES2.0.0/h5py路径与版本、case哈希和原生产attestation。`_load_reference_fingerprint`读取根公共物理元数据，不枚举事件或整文件哈希HDF5。生产模块被真实导入，未调用simulate、andes.load、PFlow或TDS。成功身份由真实FOM/spawn/GPU探测产生，仓内来源与外部解释器/模块/case身份分开记录。

九项合成测试实际调用`guarded_call/preflight/validate_identity`等生产guard；错误解释器/prefix、缺包、身份/source/case错配均在callback前拒绝，不是只断言硬编码结果。它们与真实probe共同承担角色接线证据，不证明物理数值资格或整个采集批次成功。

**唯一需在启动前补齐的清理安全问题：** `_run_probe`在`communicate`超时后直接使用`taskkill /PID p.pid /T /F`，没有核对该PID的创建FILETIME。超时可能由继承stdout/stderr的子进程尚未关闭引起，此时launcher本身可能已经退出，旧PID不可直接作为杀树授权。应复用本项目已接受的PID＋创建时间身份核验，在匹配且活跃时才杀该树；身份不明/已重用或实际probe/owned spawn清理未核实则保留未决，不误杀其他进程。该修正不要求新增科学动作或扩大三个实际环境探测/九项合成案例。

真实attestation直接依赖`experiments/closure_discovery_data.py`和`configs/private_dataset.json`，启动binding须包含两者。root通报完整binding为89个明确仓内source/config且已包含直接依赖；不需扩大到外部环境递归哈希或历史数据内容。

本地启动器复用合格PS与原native外层，实际worker身份、两层退出、清理和来源前后核对共同判定；第二次仅在原QA未通过时允许，并以首个真实原生回执扣减600秒累计上限，最多一次明确代码修正，不能将实际通过的QA重复。完整账本和修正授权由root核对。

静态决定：上述超时清理身份问题关闭且最终源冻结归档后，支持执行登记的唯一组合QA。实际PASS仍需真实退出/清理回执及FOM、spawn、GPU和拒绝路径证据；本意见不授予原48条件物理重跑。条件复用政策与完整科学预算另行登记，旧96次失败和原档关闭不追改。

## 同轮最终清理修正复核

仅静态复查最终`_run_probe`及直接binding守卫。环境模块SHA256为`b99d2257b3898f55f15bfb6e7c3647f5bc8c4f4bd2f3fee0c3d6856d0e45f7f3`。原直接按旧PID调用taskkill的分支已删除，改为复用合格原生PS与`fixed_edge_pair.native_job`；实际probe worker的PID/创建FILETIME/解释器与两层监督记录相互匹配。只有两层真实退出、非timeout、清理完成和worker已退出才接受；wrong-role允许真实非零拒绝，但同样必须监督合格。probe子包络明确包含在总QA费用内，不另加收费。

binding必需集合已包含`closure_discovery_data.py`和`configs/private_dataset.json`，以及新环境模块/测试、真实生产入口、新PS和原native依赖。原清理阻断关闭，未发现本次有限复查中的剩余确定阻断。最终静态意见为：源/命令冻结归档后，支持登记组合QA实际执行；其结果尚未知，不授予物理重跑或科学采用。此次仅读取源码和哈希、编辑本审查报告，没有执行QA、导入或读取数据权重。
