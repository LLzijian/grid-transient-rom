# 记忆幅值调制边耦合运行模块

可选ERuntime在N7上增加一个共享、有符号系数a_E的边储能：H_extra=a_E*Σ kbar*(qi+qj)^2*[cos(phi+D*delta)−cos(phi)]。kbar沿用现有编译器固定边权归一化，q沿用潜坐标；无新增端口、投影或尺度。它是原研究计划事前列举的固定机制，不改变默认模型。

令r=q*abs(D).T、c为中心化余弦，新增梯度为g_delta=−a_E*(kbar*r²*sin(phi+D*delta))*D、g_q=2*a_E*(kbar*r*c)*abs(D)，其余两块为0。两块都经原J−R与M变换进入场；给定外部u时也保留。原点新增梯度/Hessian为0，有符号储能不提供正定性或稳定域结论。

模块为experiments/edge_memory_amplitude_runtime_20261021.py，提供ERuntime/build_E/enable_runtime/extra_energy_components及原encode/field/rollout/export/from_export接口。按term_key识别新增系数，使用标准表达式/双系数副本导出。每个优化图内合并公共矩阵乘积并保留对系数、骨架和J/R的完整梯度；图结束、异常、load/device变换清空缓存，活动图内拒绝导出。

2026-10-21实际QA01为**47通过、0失败**，其中23个runtime CPU/CUDA检查覆盖generic/NumPy与两解析梯度、原点Hessian、非单位M、外部u、完整机器重编号/配套D-phi边反向、完整5s目标及全梯度/clip/Adam、非空Adam与CPU/CUDA/NumPy RNG文件恢复、真实不等长/n=1、一次初始化、缓存/非有限/导出。field rtol2e-11/atol1e-11，loss2e-10、grad/Adam2e-9；独立有限差分使用eps1e-6及近似容差2e-8/2e-9。

**科学执行未通过准入。** 唯一cost入口在干净的`-m`进程中先调用baseline_contract、后设置模块路径，导致导入symbolic_closure_rom失败。pytest预置路径下的47项通过未覆盖这一入口缺口。按已登记不重试规则，该轮关闭：无真实成本试算、三源零更新检查、训练或正式评价，没有预测收益、真实成本、加速、模型采用或确认结论。研究runner及失败入口留在归档，不在本次工程集成范围。P8整包QA失败与本轮QA通过/入口失败分别保留。

证据位于归档分支codex/edge-memory-amplitude-20261021及results/edge_memory_amplitude_20261021。源4c91237fcea9ae00941f3fabd4edffb7a2199f61；QA原生54.4136333秒（pytest48.37为子集）；成本入口失败4.8687917秒，344源一致、两层exit1且GONE。两类费用均含子层一次计账。

复现依赖主分支已有计算模块/fused audit和冻结合成资产。FUSED_CLOSURE_FIXTURE需指向本地private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01，manifest SHA0a1f2463693b0a5227c8f39e725a4e8891af0300dfccdfde705d66e346ee9816、data SHA876f05203ba4ebbf0b503e0fa8cac077e22d1f37a779d95f1feca2047f5bc04f；NPZ/RNG本地保留并记hash，main不承诺公开clone自足CI。独立复现与后续研究需另有范围/预算，不据本说明重启已关闭cost。
