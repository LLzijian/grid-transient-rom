# T4_0001 工程接受记录（不是预测成果）

## 独立复核与验证

主实现由root及modeler/evaluator代理完成；readonly_review代理未参与主实现，只读核对数据/单位、数值分类、NumPy公式、缓存和预算边界。其结论支持implementation_only接受；未宣称预测提高、公式压缩成功或稳定域。

- 完整工程测试：logs/run_6178030899a9/process.log，114 passed；相应源快照为该目录source.zip。
- 新增RK4加速组合的回归：logs/run_24df4d31c6fe/process.log，6 passed。保留独立eager函数/参数/输入梯度和RK4对照。
- 真实TRAIN预训练：P_anchor完成500步、2754窗口、346事件，73.515进程秒；动力学未在此阶段优化。
- P_short首次联合运行195.016秒后，由root为执行器QA明确停止已核验的子PID68204；保留完整step10的last.pt及全部后来步日志。退出码4294967295表示这次计划性终止，不是模型NaN。未保存的更新不冒充可恢复检查点，重算成本照常计费。

## 已接受的实现范围

严格源只读/100ms/10ms契约；解析梯度与独立autograd/有限差分核对；真实前缀分组联合优化；周期完整检查点与异常态分离；反传安全deadline；完整LOS/window-settled/数值未决与原始分母；逐事件失败隔离的固定8s评价和缓存；TRAIN-only锚缓存；保持support的有限函数压缩；完整独立NumPy显式ODE/GRU初始化。

模型晋升仍为none。完整科学路线、T4-S真实自主精调、P/A/U开发评价、最终稳定域均未因此完成。

后续分块与稀疏执行QA：run_13b995b9ce29中127 passed、2 skipped（本机无CUDA）；仅新增日志比较测试的1 ulp浮点严格相等失败。测试采用1e-14相对容差后，run_92fe1d4e4cd6四项optimizer测试通过，含新分块与旧flat两种检查点的精确恢复、Adam更新一致性及shooting独立学习率。readonly_review发现的shooting误继承coupling尺度已修复；所有当前正式配置仍为各块尺度1。稀疏支持及缓存、U共享GRU初始化均有测试覆盖。

后续尚待运行的接口：严格最终D、固定端点Jacobian/候选V、显式导出包与能力比较。独立只读复核已检查D/端点/导出；导出源哈希改为加载前、加载后及完成后核对。只有相关QA通过才记录为可接受实现，不以静态复核替代执行。

新接口QA已执行：run_e1db173f90fb为180 passed、2 skipped，唯一失败为旧合成夹具仅提供delta_star/omega_s而无as_dict接口。comparison_context将缺失物理接口标记None（真实Physics仍完整导出），run_ea55cae378e4针对完整评价回归通过。上述180项包含最终D、端点诊断、独立公式包装、完整比较器、N/S原点复用及连续优化更新回归。比较器独立复核要求的时延集合与真实n/common_n约束均已补齐并有反例测试。实际原点复用成本/逐梯度对照run_354fb03edfd4，82.719秒，equivalence_passed=true；详细数值见成本登记及JSON。

root在首次续跑时漏写P_long的reuse_origin_terms开关。已核实原进程实际使用旧执行器，停止子PID38308，保留run_f93f20b15ac7及30.781秒成本，再从该目录完整last.pt恢复至P_long_cached。未将旧执行器误称加速结果，未重置科学步骤或账本。

## 执行器有限比较与决定

T4_0001_executor_probe.json基于同一完整step10检查点、同一样本、double精度、同损失与全部梯度。eager+checkpoint10与TorchScript+无checkpoint在1s的反传分别23.36与7.625秒；损失均2.791037531439027，模型梯度L2均约1.833986882940203。短段采用后者，属于执行策略组合的成本下降，不是部署同精度速度声明。

TorchScript+checkpoint10在该安装版本不兼容，失败记录run_fc17179bf4bb、run_def92b43641c、run_c878250e04f3保留并计费。该组合已明确拒绝，运行接口主动报错；没有改变模型/物理精度来掩盖失败。长段仍由8GiB监管器约束；必要时恢复eager+checkpoint的等价执行。

恢复允许变更仅device/checkpoint_steps/trace_field这些已验证执行选项；优化器/随机状态/步数课程沿完整检查点恢复。科学参数变化仍需warm_start与新阶段登记。
# 追加记录：执行中的日志解释与待验收模块

2026-09-30，P_long_cached继续固定400步课程。steps.jsonl中的块级`parameter_l2`按本步具有梯度的参数子集累加，故已退出损失的head/shooting可记录0；这不表示存档权重被清零。科研报告不把这些0解释为模型剪枝或复杂度降低。最终S的autonomous_only会明确将head移出优化器及独立导出，二者需区分。

冻结速度核验和压缩脚本的有限输入算术溢出留痕已经独立只读复核，原阻断已关闭；新增合成测试待当前训练结束后受管执行。新结构贡献审计及原13目标模态匹配仍在实现/复核中，未记为通过或产生科研结果。

2026-10-07恢复后的实际QA：run_6564b323652a中53 passed（压缩pipeline、速度核验、原13模态、结构审计）；run_a7d006a02568中37 passed（模态配置/env源路径修复、完整评价生成时cache/meta SHA与审计预解包比对及有限失败回归）。readonly_review_resume未参与实现，只读对照原KIKI匹配公式及独立J/R/energy路径，支持implementation_only接受。当前接受代码锚为logs/run_a7d006a02568/source.zip，已记录状态历史；科研模型指针保持null。之前“待QA”的新接口已完成上述验证，真实预测/模态/稳定性结果仍未取得。

真实C已完成：固定P400，5714个TRAIN编码/教师自主/有限扰动状态，七模块各16步真实短段适配，全部选择degree2，544个非零标量系数。最终固定教师探针向量场相对RMS差0.0460738851、能量梯度0.00385910157，均有限；不是自主轨迹误差或有效域证明。run_bb0acfd71a45计575.781秒，未读取开发或TEST事件。随后执行3步严格完整联合5s成本探针，正式D仍从C终点开始。

S正式D中途诊断（2026-10-07）：第29步[ pilot_0028, expansion_0008 ]有限loss=1906.233691、原始gradient_norm=289612.1634，交换块主导；邻近步骤回落，未据此判为刚性或更改单位/门槛。原始梯度与有限实际更新均保留在S_final/steps.jsonl。第100步固定12事件TRAIN面板完整，single_start_mean=0.5399347363；P400同面板为0.5529987002。两者目标最多5s，短事件按真实前缀，使用相同TRAIN尺度与物理损失；均numerically_verified=false，不能解释为正式预测提高或据此改400步固定端点。

S正式D已完成400步，覆盖全部346可用TRAIN事件，辅助项全部退出，run_bca7bcaaaf15正常退出并计11071.437秒。固定TRAIN面板S200/400为0.5065790050/0.5025679851，仍未数值认证。区间AD、缓存独立复算及正式步长比较器的57项针对性QA通过，run_48c78908cd77计19.984秒；独立只读复核者支持条件implementation_only接受，没有科研模型晋升。S400公式包导出run_fc491cc193f5计3.156秒，固定合成Torch/NumPy最大初始化/场/梯度/能量差约3e-15；真实轨迹能力、原13模态和区域证据仍待固定端点实际执行。

固定S公式的复杂度报告新增16项合成QA通过（run_5cdfcf89eb75，3.125秒），独立只读复核已确认AST规则和完整50行因式RHS、中心化、两端口及governor没有重复或漏项。实际固定JSON统计run_4d65f1943c7e计2.125秒：7模块、34标量输出、544非零系数、810稠密槽、2772共享多项式AST树节点及4194 GRU/投影参数槽。AST不含feature、导数外层、图复制或整RHS；完全分配律展开及全RHS节点未取得，不冒称完整展开计数。导出JSON只保存coefficients*support，无法由最终零系数反推冻结训练支持，相关槽数明确None。当前未用这些工程结果晋升科学模型。

来源冻结接口及消费者71项合成回归通过，run_5e2a92c59d5b计19.750秒：checkpoint/M3/manifest SHA在开发读前核对；源改变、删除、不可读均令原JSON失效；消费者拒绝明确invalid，旧无标记仅兼容；实际张量dtype、固定buffer、共同physics/scales、short200/long400及run lineage待实际all5冻结。独立只读复核支持实现层接受，代码锚更新为该run/source.zip，模型指针仍null。U_short原始summary沿用旧generic T4-N文字，实际kind=U-control，不赋予结构资格；U_long仅修报告stage为向量对照无结构/符号声明，动力学及协议不变。

实际全5端点冻结已在run_7744a6e40221完成且独立复核通过，历史P_anchor单run源码归档缺口如实记录。M3真实开发JSON已复核，误判稳ID为pilot_0155与expansion_0155，两者数值合格，不能靠改数值或LOS门槛消除模型损害。P-N首个慢eager尝试停止并保留2批缓存，不产生终点结果；只改变执行器后同冻结权重重跑。

可选评价trace接口默认关闭，M3+trace在来源读取前拒绝；端点身份检查之后、事件加载之前trace，原参数别名与state_dict保持；没有启用原点复用。run_e7ceae977eb7中67 passed/18已解释TracerWarnings（37.71秒测试、40.156秒总账），覆盖N/S/U、B1/B8、两正式步长、全部状态/子步/物理及finite标志、初始latent NaN与延迟latent失败后缩批。第二代理静态复核无阻断，支持等价执行接口使用，不赋予真实预测/速度资格。实际P-N重跑新输出路径，不覆盖失败缓存或改选候选。

实际S区间报告与各半径区间数据已由未参与实现的第二代理只读核对，.003盒对应内含二次椭球的严格P、Q下界和rho包含/不变性链一致，原8半径全部保留，源/库哈希未变。接受范围仅连续49维数学ROM、依赖明确区间库正确性条件；不是整个盒稳定、FOM保证、50维点渐近稳定或初始化误差包含。原模态N/S实际有效0/13另列，不否决L1。上述结果与未完开发矩阵分开，尚未晋升科学模型。

2026-10-08后续针对性QA：run_d2d08ed0ecd8计80.282秒，269 passed/2 skipped（77.24秒测试）。覆盖trace评价、全评价来源冻结、原点项复用缩批及异常隔离、初始化中心有理数包含核验、默认关闭的二次多项式执行器与独立显式运行时。skip分别为本机无CUDA、底层参考本身冲掉subnormal的条件边界；不声明该边界的逐值等价。第二代理readonly_review_resume静态复核上述实现无阻断。实际P-N/P-S固定开发评价仅使用已核验trace/原点项接口，未启用二次多项式执行器。

run_8fe878356d3d计22.984秒，25 passed（19.55秒测试），覆盖固定TRAIN单步二次执行器成本探针与初始化中心检查；新增跨批缓存变更反例确保在B1处理期间修改B0时，末尾哈希撤销B0中心结论且保留全部事件。probe_quadratic_training未产生候选检查点；独立只读复核确认各模式从同一固定S400权重、Adam、Torch/NumPy RNG开始。三个模式均为trace+原点复用+checkpoint_steps=0，reference指该组合中的通用多项式，不冒充原训练完整成本基线；短失稳只使用真实前缀。实际成本及完整梯度/更新比较仍待受管执行。

实际固定TRAIN探针run_98ac16b5287b正常完成，101.719秒。第一stable pilot_0000监督5s/500步，第一unstable pilot_0030仅真实.51s/51步；未读取Dev/TEST。reference/energy_only/all_poly均完成，S400前后SHA与冻结端点一致，没有保存科学候选。两快速模式各14参数张量梯度presence/finite/容差与Adam更新容差均通过；最大梯度绝对差3.55e-15/2.13e-14，最大更新差2.014e-13/4.028e-13。第二代理核对报告完整性，未独立解码NPZ复算。单步含诊断复制31.788/21.321/27.914秒，仅H执行器在本顺序单次测量最快；后续接口优先energy_only，不以一次冷/热混合顺序成本冒充稳态部署速度或精度提高。

首轮闭合补充：run_6040b0a618b2中31 passed，24.640秒，训练runtime off/energy_only/all_poly、RNG保护、原Adam连续性及相对源/不同CWD路径契约通过；独立只读复核通过。run_00fcaff07598中16 passed，10.281秒，自主N配置进入C时teacher两项.1正常且head不调用、源零latent/torque不恢复、C存档不冒充完整D。run_85c7829666d8中31 passed，22.578秒，实际子报告按真实optimizer成员、成功目标调用和已完成step梯度参与区分head作用；旧粗标签标legacy，无使用时参数不变，positive anchor不信False元数据hint。另一只读代理复核通过；最终接受代码锚更新为logs/run_85c7829666d8/source.zip，模型仍未晋升。

实际全五端点、每端点固定12缓存面板、N/S各856结构状态、S 107初态中心及17次runtime积分均完成并只读复核通过，严格范围见T4_0001_closure.md。结构JSON中independent_personnel_review=False保留，数值复算由root执行；人员独立部分为只读挑战。初始化中心均outside不是失稳。16计时trial+1同公式S Torch参考的8次S物理parity通过，N无独立parity，全状态差只是诊断、非阈值门；无同精度速度或FOM声明。

