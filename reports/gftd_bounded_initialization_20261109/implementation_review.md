# 有界核合法初始化实现审查：接受冻结源码进入准备与固定资格

**接受本报告绑定的公共实现、执行入口和资格驱动，未发现阻断准备或固定人工资格的问题。** 同一有界核已接通储能、全梯度、弱积分字典、初始化配置与模型存取；执行入口直接保存原六门中的未来数组，并为每个正式端点安排至多一次严格 Radau。真实25包小试须在准备、人工资格结果和源身份核对通过后由 root 释放；其余272包另按已冻结触发条件释放。

日期：2026-11-09。Reviewer：`/root/feasibility_reviewer`。本次独立检查源码、差异和既有公开元数据，未执行 pytest、模型 RHS/H/梯度、优化或 ODE，未读取私有观测内容。源码接受决定不是人工资格已通过，也不是模型或科学结果晋升。写作应用用户指定的 anti-defensive-writing 原则，以具体代码和协议条件界定结论。

| 冻结对象 | SHA-256 |
|---|---|
| `src/kiki/physics/kuramoto_functional.py` | `6e0c6bf1e2c1018bae2c3d7d134e2bf67775691afcce936fdb6f92b933d317f0` |
| `src/kiki/models/gftd_discovery.py` | `1ef683bf30bc1a93a6a0fd38a198c9d0add1d885b63660a4719222a0d5577685` |
| `src/kiki/models/gftd_fitting.py` | `06e34e22ea3b04961555f21247162c22d888443be452b706ba3e1f419f0f40ac` |
| `src/kiki/models/gftd_kuramoto.py` | `20f9d6f6fc83168e08ad1886e12ad1907e4e5e350c5ec6cba1342a9a6ec54e3b` |
| `tests/test_gftd_bounded_kernel.py` | `bf62fc758af178a6490165a50e7d0ffbb1af0dc9a91e880103c94c924aef659c` |
| `experiments/gftd_bounded_initialization_20261109/qualify.py` | `29c07d6fdf924095011aa980ca8d738f86782b249f2fc6e8b0bdbc7982c28b3f` |
| `experiments/gftd_bounded_initialization_20261109/run.py` | `9ea499d9d80a7990c3cc5a67e76a5ce7273c5c65940f24651002e683ef90af8b` |
| `tests/test_gftd_bounded_initialization_runner.py` | `6a6f27101f9a73440c0b90351127ac48a65ff6bcae353984067581f1e5ea629e` |
| [架构](../../designs/gftd_bounded_initialization_20261109.md) | `c6089e0a133b4d696801734ae6cc830c60db13cbfe125d3ecbf64ee829912a1b` |
| [协议](../../campaigns/gftd_bounded_initialization_20261109/protocol.md) | `019a2dc59c4c47d761baa6a3856b675fa5efafaad62fec5830c6286daad3353c` |

以上文件实际字节已逐一核对。设计接受记录见[设计审查](design_review.md)。实现相对父 commit `989fa944da3c47ec09abda768bd6eb039274eb0c` 审查；正式准备还须保存实际导入的全部 kiki 源码闭包，不能以本表替代运行绑定。

**公共方程与旧模型兼容。** 新核使用 `K(h)=exp(tanh(h))`，同时返回导数因子 `1-tanh(h)^2`。H 的边项采用 `Phi=(1-s)+s*K`；相角梯度采用同一 Phi，幅值梯度采用 `s*K*(1-tanh(h)^2)` 乘端点 gamma。RHS 继续从完整 H 梯度和原耗散算子构造。纸面链式求导与实现一致，没有只替换 H 或只替换相角力的分裂路径。原指数分支仍在原有严格浮点错误上下文中执行同一 exp 及乘法次序；默认核字段从旧 `_physics` 和旧保存载荷省略，原模型身份保持兼容。

`conditional_dictionary` 使用不含当前 k、门的 K(h) 列，求解坐标仍为 `(c-beta,beta)`，解释列仍为 `[A_static,A_K-A_static]`。弱历史缓存本身保存不依赖调制核的静态／端口项；上层 `_prepare_batch` 和 fit297 缓存键经 `_physics` 纳入新核，候选调制列按当前核重新生成。未把旧指数列当成新核结果复用。

v1/v2 继续解释原指数模型，拒绝携带核字段；v3 必须携带精确 bounded 标记，未知或缺失标记拒绝。当前因果候选的核还受嵌套配置指纹约束，删除标记并降级格式不能通过该配置校验。保存拒绝不能由版本化格式表示的 functional 子类。deepcopy、参数克隆和规范变换保留 dataclass 的核字段。人工规范变换算例局部储能系数为1，检查的是该固定算例下核身份和对应量保持；没有据此扩大为对任意坐标变换的数值验证。

**准备与输入范围。** `prepare` 读取冻结设计／协议、父模型、父594行资格元数据、原输入清单及源码，重建25包固定索引，检查297个事件顺序和267/30分母。它不读取 `inputs.json` 的观测内容，不调用初始化或模型 RHS/H。父模型经唯一核替换后另存为 v3；数值 functional、gates、encoder、delta_star 与父载荷逐字段比对，嵌套 fit297 配置除核对应指纹及 preparation_key 外保持相同。新记录包含 refit=false 和父 preparation_key 来源，不伪称新核已经拟合。

原私有包内容仅在真实阶段释放后，由 `load_packets` 按原清单路径、字节数和 SHA 读取，并核对原事件顺序、完整分母、每包字段及 sigma=0。事件ID和标签只留在任务／报告元数据；进入 `model.initialize` 的包只有 theta_history、omega_history、history_times 和 p0。原始 val/test、真实未来、故障元数据和新 FOM 不在该读取路径内。

**固定人工资格与调用计数。** 两个测试文件限定为预设单点、手工字典缓存和 mock。六个正常候选配置各做两种步长的全梯度中心差分；两个饱和配置做有限性、饱和公式和功率检查，不做差分；原静态对照为两个既定正常点。相同正常点的 save/load、deepcopy、参数克隆和规范变换检查没有新增轨迹。字典检查使用原定N=3主／细网格及零／线性幅值数组，不调用端口重建或优化。

按源码推导，若测试全部运行到正常结束，人工模型工作为14次 RHS、374次 H、14次 gradients、8次 conditional_dictionary；字典弱力对照直接调用 effective_edge_couplings 共488次。这些是预期计数，实际结果须以资格回执为准。模型 RHS/H 的方法包装在进入原函数前阻止第129／2001次调用；profile 按原代码对象计数，不抛异常，也不会把 mock 替代函数算成真实模型。已知 DP8、RK4、端口重建、初始化优化、共享fit以及 Radau/DOP853/solve_ivp 别名在 pytest 导入前被禁止；当前冻结测试没有绕过这些守卫的真实轨迹路径。`tests/conftest.py` 仅设置源码路径，pytest 配置没有额外模型 fixture。

资格驱动先核对准备中全部源哈希和两个测试文件／驱动自身绑定，再写唯一 claim；即使失败，也通过 finally 写实际调用、违反项、退出码和耗时，恢复包装。正常资格只执行一次。若之后针对明确缺陷复查，原回执与128 RHS／2000 H共同余额须先由 root 对账，不能删除 claim 后把新过程视为新额度；当前入口主动阻止自动重放。

**两阶段初始化和原数组采集。** 执行入口调用公共 initialize 一次；simple600、smooth1200及全部有限差分计入父残差计数，核对原固定 DP8 每步12次 RHS和每套六门6480次上限。没有改拟合目标、边界、选择规则或 stage2 失格后的处理。捕获包装在调用原 `_integrate` 一次后复制原未来主／细101×50数组，校验 endpoint 字节哈希、方法和数量后原值返回，finally 恢复；未改 dense 参数，也未调用会重新初始化的 rollout_state。

正式选点按实际最终阶段和端点哈希关联主／细轨迹。stage1复用时复用原检查和数组，未重新求值；stage2被选但失格时 formal_point 保持 null，stage1 单独保留，不能替代正式成功。历史往返保留原六门诊断与最优点历史证据，符合协议“至少保存未来全状态、历史保留原资格证据”的范围。

**严格参考与完整分母。** 每个正式六门通过点安排一次严格 Radau，24000上限在每次实际模型回调前检查，包含构造／数值Jacobian、step及dense_output所用回调。原样保存已完成的连续共同输出网格；未完成部分为缺失，不用末点填充。细步和主步分别与它比较，只有完整101点、Radau完成且观测RMS／peak和全状态相对差均达预登记门才确认；短区间误差可报告，整段确认保持false。额外主步确认失败不会改写原六门。

报告表保留全部297行和267/30子组；未执行为null/not_run。旧静态与旧指数资格分别保留，历史配对使用旧 `candidate.history_mse.candidate`，并标注新历史点是否正式返回点。参考确认、stage1阻断、stage2选点失格、优化／历史求值／参考失败等记录分开。进入全量的条件是25包完整回执、无未知／中断执行以及至少一个正式p0精确端点；未增加净改善、旧成功全保留或Radau通过的筛选门。全量只提交余下272包，并核验／复用小试结果、诊断和轨迹的原字节。

**异常回执与阶段释放。** 每包先写独占 claim，初始化返回后先保存计数，再写诊断／数组；参考回调计数保留在记录中，归档或后处理错误由 finally 写中断回执。无法确认的执行不会被记为完整成本，阶段回执保留已知成本、partial/unknown/unclaimed项；相同阶段和包的 claim 阻止自动重新计算。子进程串行处理自己的包，不并发覆盖积分包装。root释放文件绑定准备、资格、设计采纳、实现审查、动作ID及worker数；worker还检查源闭包、运行库、候选模型和配置身份。

接受状态为 `source_accepted_for_preparation_and_fixed_qualification`。允许执行已审查的零模型求值准备与一次固定资格；真实小试仍待资格回执核对和 root 的 pilot 释放，272包仍待小试条件核对和 full 释放。本报告不把有界核的有限时间存在性写成长期稳定，也不把一秒数值资格写成预测正确、共享参数充分适配或L1/L2/L3完成。

未执行测试或真实数据计算；下一项实际风险核查是资格守卫回执、候选模型／输入身份及原轨迹与参考数组的一致性。出现新源版本或未知执行成本时须先对账，不以重跑消除缺口。
