# 首次物理执行wrapper：执行前有限实现审查

2026-10-20，独立代理 `/root/fixed_edge_final_review`。本次只读新wrapper、专用本地launcher、首版接线合成测试及直接旧helper接口；未运行、导入、编译，未读取真实权重、轨迹或旧val/test结果。仅新增本文。测试最后一项批停止产物检查仍在静态收尾，由root封版后核对，不据未完成文件宣称通过。

**当前wrapper/launcher未发现阻止登记QA的确定缺陷。支持最后测试封版、源与精确命令冻结归档后，进入本轮有限接线QA。** 这是静态执行准入，不是实际QA、物理求解或整档完成的证明。

## 已核对的接线

- `prepare_binding`校验原design/val字节哈希及原48×2统一启动失败谱系，检查原明确short目录没有原始物理输出；不重新抽样。复制旧scientific manifest，仅迁移`design_artifacts`到新副本，保留原`verify_sources`所需兼容格式；新增wrapper/环境/命令身份由外层独立绑定承担。
- 早期`wrapper_worker.json`写入额外独占目录，acquire/fit拒绝其与科学output相同；未预建旧helper所要求的新建科学目录。launcher核对这个真实PID/FILETIME/解释器与原生回执。
- 实际生产preflight使用原合格QA完整期望身份；FOM类别使用已验收ANDES解释器，score/fit用GPU解释器。每个单条件边界再核环境/源码；`BatchEnvironmentStop`继承BaseException，可穿过旧catch Exception停止批次，局部回调在finally恢复。数值/TDS失败保留为条件局部失败，没有统一升级为环境失败。
- wrapper保存当前边界记录与剩余not_attempted；不以新一轮同错继续调用48个条件。新增最终合成case应实际核对该产物和47条未尝试，而不只测试sentinel类型。
- fit数据/预测清单/模型/输出路径resolve后限固定新档，旧val/test路径不作为输入；原366仅经许可train_cohort。fit.CAMPAIGN只在当前进程绑定到固定新BASE，并在finally恢复旧值，路径边界未放宽到整个private_data。
- 适配只接受原六个P2来源目录，明确核对latest指向登记step001024、checkpoint/model/training哈希和源identity；新stage绑定模型及Adam来源。global1280模型须在固定新模型目录，核模型和训练状态哈希；六端点与原条件仍经旧真实freeze接口检查。未修改损失、日程、样本资格、预测公式或统计标准。
- launcher绑定本身、实际command/prior、新wrapper/测试、环境模块、旧生产helper及原生依赖/协议；类别、worker输出、固定campaign、解释器和提前60秒自检有闸门。明确prior回执按类别与单模型累计，QA最多一次失败修正重验，原科学单次阶段不能静默重跑。

## root实际启动时仍需落实

1. 最终测试、完整来源/环境QA身份、精确命令和新设计副本均冻结推送；此次审查不代替源码最终哈希或真实QA。旧source_manifest、旧代码和旧96次失败保持。
2. 后续freeze-val输出的`frozen.json`必须在新`CAMPAIGN`内。旧采集`verify_val_freeze`要求相对campaign，而不是仅相对BASE；精确命令不能把freeze输出放到BASE的其他同级目录，再到生成val时才发现不兼容。
3. 这是root受控的单次launcher，不会自动证明此前回执列表齐全或所有作业均退出。每次启动前核实际PID、无重复、全部失败费用、类别及单模型余额；全阶段唯一外部中断恢复、最多31步重放须据真实事实授权，不由单模型目录或CLI flag自行产生许可。
4. 原生成/评分阶段exit0可表示已完成失败记录或D不足分支。root须再核科学status、合格数量和原门槛，不能把原生正常退出当作下一科学阶段准入。

最终科学解释仍限同一预锁条件首次物理求解和固定闭合结构比较；工程集成不采用模型，不开放原test、旧val或TASK4，不重新标记旧环境失败、HDF5字节读取违规或P2的512次未决。
