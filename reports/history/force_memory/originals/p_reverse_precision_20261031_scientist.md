# p_reverse_precision_20261031：反向 p 传播精度探针

**目标是解释已失败的一致性门及其对实际因果初始化的影响，不追认旧Run通过。** 本Run为probe，固定A96与A_30311 step16；optimizer=0、共享参数拟合=0、新ROM rollout=0、新FOM=0。只在丢弃的内存副本中对照p_backward，原模型、候选与数据全部不改。执行必须等p_port_diagnostic_20261031报告/独立审查/选择性提交及远端归档核验完成，先设计不越过依赖。

2026-10-31；task_id=`/root/scientist_checkpoint`；既有spawn配置 `requested_model=gpt-6-astra`，`server_resolved_model=null`。本次仅写设计，未执行探针。旧门atol=rtol=1e−12的失败、M4原buffer资格0/297、M3解释unknown均保留；新诊断不能修改旧协议/结果，任何资格修订须另行前瞻登记与独立审查。

## 1. 已知与待区分解释

已有加载buffer与SciPy重建最大差1.56730184e−10，历史路径差最大4.60337e−9；原M1模型自身轨迹门通过，M2为混合来源。这些数值既不证明p_backward无关，也不证明物理结构错误。上游tm的转矩标注仍按原来源说明限制解释，不转换p、不称真实SI功率/能量。

四个解释分别检查：E，矩阵指数算法在同一精确输入上的误差；D，构造dtype/中间乘法或JSON转换；P，buffer生成时与当前g/a不一致；I，反向符号、块排列或实际调用有实现缺陷。允许混合或unknown，不以最接近的一种计算方式自动认定历史因果来源。

## 2. 唯一输入与只读边界

沿前Runmanifest固定A96与A16的model/源文件哈希及加载器，核g=jpi_p、a=rpp*storage_base、p_backward、mass、p_scale、rstar和真实dtype/device；保存Python/Torch/NumPy/SciPy及高精度库版本。只读A96原缓存与checkpoint_selection的A16完整回执绑定缓存；仅取NPZ的event_ids和histories成员，**不加载fine/coarse预测或任何未来真值**。

从histories提取11点δ/ω和末点p₀，共297个相同ID；两份合法历史逐值核对。初始化传入的历史p前10点置零占位、末点保留真p₀，以免意外使用历史p真值。保留所有事件，不按标签挑选，不读val/test、train385/noise88，不调用原read_train。前Run所有未来残差只引用已公开结论，本探针不重算。

固定CPU float64；不比较GPU、不更换设备谋取过门。两模型常数/生成矩阵完全相同则数学核查去重一次，初始化仍按两个不同模型分别做。旧源码只读，新增诊断实现/结果由root在新Run目录登记。

## 3. P0：先确认实际调用，禁止把活跃buffer当历史附件

读取并冻结 `ForceGraphROM.__init__/from_export`、`StateGatedMemoryROM.initialize/encode/history_system/from_export`、`MemoryTransportROM.history_system/from_export`。当前源码显示两种history_system的反向p循环均使用self.p_backward，且initialize→encode→history_system；这只是静态证据，须核加载对象实际类/方法解析顺序/函数源码哈希。

在P2规定的原buffer初始化调用中，使用只读函数/行调用追踪记录实际initialize、encode、history_system入口及含p_backward矩阵乘法的源行命中；保存绑定方法和buffer值哈希，不重新实现初始化替代真实方法。可用Tensor运算追踪补录其存储地址被转置/矩阵乘调用。预计每次history_system有10次反向p传播；实际不符先定位调用路径。

有源行实际命中及该路径产生p_history/r₀，记buffer活跃；若最终源码和运行追踪均证明真正初始化未读取此buffer，记dead_buffer并说明实际替代路径及其一致性，跳过无意义的替换。追踪失败或输出差为零本身不证明无用，记use_unknown。不得为证明使用而注入大扰动或改其他参数。

## 4. P1：同一输入的独立数值参照与精度来源

按加载后float64常数构造原30×30 P：块P_pp=−diag(a)、P_pω=−diag(g)、P_ωs=I；先在float64形成 **Q=−0.01P**。固定比较的主数学输入就是Q的逐元素二进制float64值，防止把输入舍入与expm误差混为一项。

主参照：将Q的精确二进制值提升到80与120十进制位，各算一次高精度expm。另按每机3×3上三角块独立解析，若该块为[[d,b,0],[0,0,c],[0,0,0]]，首行为 `[exp(d), bφ₁(d), bcφ₂(d)]`，φ₁=(exp(d)−1)/d、φ₂=(exp(d)−1−d)/d²；d=0直接用φ₁=1、φ₂=1/2。解析也用高精度，不能先float64相减再提升精度。

80/120位矩阵结果及120位解析结果均须满足 `maxabs(diff)≤1e−60*max(1,maxabs(reference))`；不满足则reference_unresolved，停止以它替换buffer。人工核查只用前Run固定零a/常斜率式，增加的exact-zero项仅验证公式，不生成任何FOM。

固定低精度组合每个唯一Q各一次：①现加载B_old；②torch.matrix_exp(Q_float64)前10行；③scipy.linalg.expm(Q_float64)前10行；④Q_float32的torch.matrix_exp再转float64（仅dtype指纹，不作为修复候选）；⑤高精度参照四舍五入至float64的B_ref。不扫算法选项、精度档位、容差或缩放阶数。

另一次记录：JSON十进制token→binary64→加载tensor的逐位差；用80位从原rpp/S/g的精确binary64重新作乘积及Q构造，量化“常数相乘和Q装配舍入”相对主参照的影响。该项不替代主输入。核导出config与有效state_dict的常数、原构造代码及能够定位的祖先checkpoint哈希；只查已有谱系，不枚举新模型或恢复旧运行环境。

所有组合报告max绝对误差、Frobenius相对误差、逐块误差和ULP差；参考零项的逐元素相对误差写null并给绝对值。报告旧门的最大标准化违约 `max |Δ|/(1e−12+1e−12|ref|)`，不换旧门。当前Torch64若重现B_old且对高精度偏离，支持“当前同算法可复现的expm误差”；历史版本未绑定仍不能断言生成时唯一原因。

若JSON加载逐位相等则不支持JSON精度损失；dtype32指纹近似不足以单独证明曾用float32，必须有构造/存储证据。若相同Q的各可靠实现相符而B_old不符，优先查旧参数/序列化谱系；来源缺失留unknown，不反求最贴近B_old的参数。任何“同值”声明均列明确误差和判据，不能仅看打印位数。

## 5. P2：真实因果初始化的有限敏感性对照

仅在P0活跃、P1参照合格时执行。固定组合为2模型×2个buffer（原B_old、仅替换B_ref的内存副本）×2个历史步长（原主.0025、细.00125），共8次对全部297合法历史的批量initialize。允许固定内存分块，科学组合不增。每次空缓存开始，no_grad；包装真实history_system捕获已有返回p_history/r₀/design/bscaled，避免为报告再跑一次。

替换副本只copy_ p_backward并清缓存；核除该buffer之外state_dict逐位相等、全部参数requires_grad/训练状态不变。initialize的末点δ/ω/p须原样相等。原对象前后哈希不变，副本不export、不保存为候选。因果r₀线性求解属于原初始化，不计共享参数估计；不得调用rollout（包括steps=0捷径）。

逐事件及每历史步长报告Δp_history的绝对max/RMS和/p_scale归一化RMS，Δr₀的绝对max/L2和/rstar归一化L2，design变化、Δbscaled与QR解残差变化；参考范数为0的相对误差填null。给297个事件的全表及max/median，不只列均值。两步长各自原/替换对照，不把历史积分差冒充buffer差。

利用同一线性递推核对 `Δp_left=(B_ref−B_old)v_old+B_ref,pp Δp_right`（ω/斜率相同）；从末点Δp=0向过去传播10次。实际与递推差须atol=rtol=1e−12。design若不变，再用原QR算子将Δbscaled传播为Δr₀，核对相同容差；不成立即实现/路径未解释，不能宣布“只是expm”。这给出小矩阵误差是否被历史斜率或反演放大的直接证据。

前向影响仅报告**初始化输出与初始瞬时导数**：在两初态各调用一次同模型field，不积分；同末点δ/ω/p应有Δωdot=−Δr₀/M、Δpdot=0。用原字段核对并报告归一化/绝对差，不以一阶外推冒称8 s风险变化。有限8组初始场读出不构成新ROM rollout。

## 6. 判读、停止与后续条件

主判读分为“误差来源可解释/未解释”“buffer实际使用/不用/unknown”“输入输出敏感性已量化/未决”三轴，不设一个小数门来宣布科学无关。若P1可复现expm偏差、P2传播恒等式通过，结论是该差异的产生与传递在登记范围可解释；仍报告实际Δr₀/Δωdot，不能由误差小推论旧M4的全部历史数据不相容已消失，更不能解锁旧M3。

若dead_buffer有完整证据，旧针对该buffer的失败仍保留，但标清它不能独自否定实际初始化一致性；实际调用路径需本Run已有数学参照核查，缺参照则unknown。若参数/符号/块或实际路径不符，记录具体实现问题，另行有限修复设计；本Run不修改模型。

有限边界为P0一次路径核查、P1固定组合、P2至多8个初始化组合和对应初始场；无参数搜索、无额外种子/模型/事件。非有限、身份改变或高精度参照不一致停止依赖项并保留证据。纯诊断实现错误至多一次修复并重跑受影响部分；门失败本身不触发修复或放宽。记录读取/参照/初始化/总wall；中断先对账，未知完成状态不盲目重放。

代码可retain_branch，模型retain_evidence/no_change，科学限定为数值来源和100 ms初始化敏感性；实际参数/物理机理与长期预测影响仍未由此确认。下一步若拟采用解析buffer或修订资格，必须新前瞻协议、独立审查、完整原因果/数值/关键事件验证；原p_port_diagnostic结果及1e−12失败均不回写。
