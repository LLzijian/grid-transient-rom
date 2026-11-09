# p_reverse_precision_20261031：独立审查

**设计审查通过，可进入实现冻结与核查；真实执行仍待冻结实现审查。** 三条证据链分别回答数值差如何产生、p_backward是否进入实际初始化、该差异如何传递到初值与瞬时导数。当前设计不会用新的参照或小幅敏感性追认父Run通过，也不会由初值差推断长期预测影响。

2026-10-31（Asia/Shanghai）。task_id：`/root/review_checkpoint_audit`；role：Reviewer；configured_model：`gpt-6.1-sol`（root任务合同），server_resolved_model：`unknown`（工具未提供）。报告按项目要求应用anti-defensive-writing，结论限定于已读证据和当前授权范围。

本次只读冻结设计及当前提交信息，未读取写入中的实现、私有模型/历史/预测、未执行初始化、field、矩阵参照或训练。旧p端口诊断审查报告保持冻结不动；本报告帮助root判断这个独立probe能否进入执行准备。

## 1. 冻结输入与依赖

| 实际读取的输入 | SHA256 |
|---|---|
| reports/p_reverse_precision_20261031_scientist.md | B3BE4D5DCA56E13123A05279975D6B08B808C158BB664700525729D7512B5D93 |
| reports/p_reverse_precision_20261031_protocol.md | C93D0443F57FEEDD75D7B1E189510BCA1364B6222B4F17D7DF715864EC968C74 |

两者与root交接值一致。实际读取的HEAD为 `f48987abcc10f5361194ca96419a197f62fa0269`，与root报告的父Run已提交/推送并核验远端提交一致；本Reviewer此次没有另行读取私有archival_verified或重复查询远端。

父Run的1e−12反向块门失败、M4计数和M3 unknown保持原记录。新probe是独立数值问题，不是父Run第二次技术修复、重跑或资格回写。上游tm的转矩标注及p冻结数值坐标解释边界继续适用，不增加物理参数或能量识别主张。

## 2. 科学与数学设计核查

**P0实际路径。** 设计同时要求静态方法解析/源码哈希和真实initialize调用中的源行命中，捕获原history_system输出，未用重写初始化替代实际实现。源行实际命中和输出路径可支持buffer活跃；输出差零或追踪失败均不自动说明无用。dead_buffer须有真正替代路径及一致性证据，不能为了简化解释跳过使用审查。

**P1输入与参照。** 主比较固定先在binary64形成的Q=−0.01P，提升的是Q逐元素的确切二进制值。80/120位矩阵指数和120位独立三阶块解析采用明确的1e−60相对/绝对混合尺度核对；独立解析首行exp(d)、bφ1(d)、bcφ2(d)与该上三角Q的块结构一致，零d极限正确，不应先以float64相减后才提升精度。

固定Torch64、SciPy64、Torch32指纹、原buffer及高精度舍入buffer各一次；没有选择最有利算法、阈值或精度组合。原门标准化违约直接保留旧atol/rtol。JSON转换与常数乘积/Q装配舍入另列，能够分开算法误差与输入误差。落实该同输入原则时，装配对照中的时间步0.01也应按原binary64精确提升，不临时改用精确十进制0.01后把两种时间常数混成纯装配差；这属于现有同输入合同，不增加测量。

历史生成原因的解释范围适当：当前Torch64重现原buffer且偏离高精度只支持当前算法可复现该误差，不独自证明过去版本的唯一生成原因；Torch32接近仅为指纹，不证明旧模型曾使用float32。原常数/config/可定位祖先用于既有谱系核对，未知保留unknown，不反求最贴近旧buffer的参数。

**P2传递与瞬时影响。** 2模型×原/参照buffer×2历史步长最多8组全297事件真实initialize，实际调用中捕获既有p_history/r0/design/bscaled，避免重复跑初始化。只在丢弃副本copy_该buffer、清缓存，其余state_dict逐位相等且原对象前后不变；不export候选，不调用rollout或steps=0捷径。

同一omega/斜率下，Δp_left=(B_ref−B_old)v_old+B_ref,ppΔp_right是两个递推相减所得恒等式，末点Δp=0边界正确。若design不变，用原QR算子传播Δbscaled到Δr0，可检验历史差及反演放大；若design或传播不符合预期则保留路径未解释，不能直接称只是expm。每步长独立对照和逐事件max/median避免把主细历史积分差混为buffer差。

同末点delta/omega/p的初始field差应满足Δomegadot=−Δr0/M、Δpdot=0，与现模型方程一致。仅给初值和瞬时导数，不积分、不用一阶外推声称8s风险或分类变化。有限8组初始field不构成新ROM rollout，也不等于新增共享参数拟合。

## 3. 输入权限与有限执行

输入固定为A96/A16原绑定模型与完整缓存，只解释NPZ的event_ids/histories成员，不加载fine/coarse或未来真值；历史前10点p置零占位，初始化只使用11点因果delta/omega和末点真p0。全部297事件保留，两份合法输入逐值对齐，不按标签挑选。

CPU float64、固定高精度组合与最多8初始化组构成有限工作量；常数/Q相同的数学核查去重，但两个不同模型的实际初始化分别检验。无GPU比较、无参数搜索、无新事件/种子/数据、optimizer=0、参数拟合=0、新ROM rollout=0、新FOM=0。原val/test关闭、TASK4暂停、default none及所有指针不变。

P1参照不合格、实际路径不明、非有限或身份变化停止依赖项；一次纯实现修复与不确定完成状态先对账的规则明确，旧门失败不是修复或放宽许可。wrapper日志与独占输出目录分开，避免父Run首启动目录冲突；成本和退出回执先登记、原生进程身份只留私有，未借此次probe重置父Run预算。

## 4. 当前审查决定

未发现进入实现阶段的设计逻辑阻断，也无需额外训练或无界确认才能完成这个问题。实现审查应核对精确Q提升、参照资格、真实调用追踪、组合计数、内存副本隔离、恒等式及不可重放记录是否落实；目前不声明这些真实检查已通过。

设计阶段三个建议决定：代码no_change（尚未审实现）；模型retain_evidence/no_change（不采用）；科学continue当前数值probe准备。最终科学结论只涉及误差来源、实际使用与100ms初始化敏感性三轴；长期预测影响与物理机制仍未知。若以后拟采用解析buffer或修订资格，另行前瞻协议及原因果/数值/关键事件验证，不能由本probe自动授权。

## 5. 首版实现预检与启动前修订

本次追加前本报告SHA256为 `DDDD609661E683692831164630294F6C310EDB8D30865B2CA54C19980CCB7F7E`。本Reviewer只读冻结源码和人工测试，未读取私有模型/历史、未执行本Run。以下均为真实启动前的候选代码预检，不是已执行Run的修复、重跑或预算重置。

| 阶段 | 实际读取源码SHA256 | 实际读取测试SHA256 |
|---|---|---|
| 首版冻结候选 | 3DD6ADB1CA50BE19EE814E158895F264985ECE5D9701478E4C80BB6BB7ECE6EA | D53B576C0C867FFEFE7492B5C0476BD1AF20D7CFC57555778B6861B56DA83DD7 |
| 两项阻断修订版 | A9EA12FF3754D63FE297A51858CD7981A421DB1D54CF1DF871CA47398C100D54 | C7F4421E05FB6165FE371A53577E32CA28E53D7346EDEEB6B346530D435C1627 |

首版发现两项具体问题，已及时告知root不要放行：

1. 原第407行main guard位于call_source等helper定义之前；直接CLI run会在run第340行调用尚未定义的call_source而失败。import测试会读完整模块，不能排除该执行顺序问题。修订版将main guard置于所有函数之后的文件末尾，并增加无私有载荷CLI --help及AST末尾顺序检查，问题关闭。
2. 原load_sources使用无后缀archival_verified存在性marker，未与协议所列真实archival_verified.json一致。root确认真实JSON的local_commit与remote_commit均为f48987abcc10f5361194ca96419a197f62fa0269、ordinary_push_verified=true。修订版读取该实际JSON，严格核两commit及布尔is True，并将实际proof哈希绑定manifest；字段缺失、错commit、false或数值1均不能通过，问题关闭。Reviewer未打开私有proof，确认的是代码合同及root既有记录路径。

root记录两个候选均未实际启动，原生启动/科学工作为0；该启动前源码修订不消耗本Run一次真实执行后的技术修复额度。Modeler报告首版9项、修订版16项人工测试通过；本Reviewer已读取测试，未把该回执称为自己重跑的结果。

其余静态核查未发现启动或科学逻辑阻断：实际Q与时间步按binary64精确提升，80/120位及独立标量块解析核对、固定低精度组合和零项相对误差null处理明确。高精度expm按Q的相互独立3×3块计算，等价于置换后的块对角矩阵函数，数学输入仍为同一固定Q，没有新增参数搜索。

P2的两模型×两步长×原/替换buffer最多8次真实initialize及8次初始field；capture包装原history_system取得已有输出，不重复调用初始化。p递推、固定design上的QR差传播和初始field差公式与设计一致，额外QR线性传播不是共享动力参数拟合。副本只替换p_backward并清缓存，原state_dict哈希前后校验，无rollout调用。历史只解释event_ids/histories成员，前10点p清零，fine/coarse及未来真值不解释。

实现一致性建议是相同原g/rpp/S的precise_assembly_effect也去重一次；两模型JSON及祖先来源仍须各自核查。root已接受并要求最后小修后重冻；本阶段暂不发真实执行放行，待最终源码/测试哈希复核，不扩大科学测量或调整门。

## 6. 最终实现复审与执行决定

**最终冻结实现审查通过，可执行当前有限probe，无剩余实际启动阻断。** 本次追加前本报告SHA256为 `CAD99A628C1086FA946C9C19730E51D2C99D9DAD789A68EFBCECD6F11E8CF7A5`。

| 最终实际读取的冻结输入 | SHA256 |
|---|---|
| experiments/p_reverse_precision_20261031.py | 6B4E2057CCAE8FE53EF71B59BCAE1EAE310C24623E6481AEF9C0635ADC9AA5D7 |
| tests/test_p_reverse_precision_20261031.py | 7973ACC32A5CF2220FEBB1F86230B1DE9CC4EC07D1D955E0BABE6EA41D30F088 |

两者与root指定最终值一致。源码末尾main guard在所有helper之后；实际archival_verified.json读取、固定父commit的local/remote严格比较、ordinary_push_verified is True和manifest proof哈希绑定均保留，§5两项阻断继续关闭。

cached_precise_assembly按原g/rpp/S三个binary64数组各自dtype/shape/bytes的摘要联合缓存。相同三数组只计算一次；即使两个rpp/S组合产生同一舍入a/Q，原因子不同仍分别核装配舍入，避免把不同输入误合并。主Q高精度及低精度组合继续按Q去重；JSON token与祖先证据仍逐模型核对。新增人工测试覆盖相同数组一次和同Q不同原因子分别核查，未扩展科学组合或改阈值。Modeler报告最终17项人工测试通过，本Reviewer已读取最终变更及测试，没有重复执行或虚称独立运行通过。

实际可执行范围仍为P0路径追踪、P1固定数学组合、P2至多8组297事件真实初始化及8次初始field；每次直接捕获原history_system，不再为报告重复初始化。差传播重算QR是固定初始化算子的线性敏感性验证，不是共享参数估计。只在丢弃副本替换p_backward，原模型state_dict哈希不变、前10点p占位、只用末点真p0，不加载未来成员，不调用rollout。

当前决定为代码retain_branch（已审实现）、模型retain_evidence/no_change、科学continue当前probe。真实参照资格、实际路径及递推/QR/field等式的通过状态须由本次执行记录证明，当前不提前宣布。旧父Run门失败及M3 unknown不回写，不据较小初值/导数变化宣布长期预测无关或授权采用新buffer。

本Reviewer未读取私有输入或运行真实probe。root仍须以这份最终源码哈希登记唯一真实启动并保留成本/回执；§5的启动前候选修订不构成真实运行修复次数，真实执行后的有界修复规则照协议保留。

## 7. 冻结公开结果与执行回执核查

**结果审查通过：在固定Q与当前运行环境中，旧buffer差异可由当前Torch64矩阵指数复现；buffer实际参与初始化，其对初值及初始场的传递由登记恒等式解释。** 这不是历史唯一生成原因的认定，也不确认长期预测影响，更不解锁父Run的M3或改写旧门失败。

本次追加前本报告SHA256为 `46F58CC69F04EF3446A7FBE411019DD4E8F69164DDEC6C62CE72A59592A7110E`。此次仅实际读取以下公开结果、核字段及冻结源码/设计哈希，没有重算私有输入、初始化、矩阵指数或预测：

| 公开结果 | 实际SHA256 |
|---|---|
| results/p_reverse_precision_20261031/public_probe.json | 7BCA3C27371433E8B2C51EE15AA9EB539382FB0CD96B761FF066E73CA3816B13 |
| results/p_reverse_precision_20261031/execution_receipt.json | 933D59C10E27BCCC941B58CDC7070365A013928E7A4355CC92EB7E55D6D37DD2 |

manifest摘要为 `270C8E8B44F8EEC4BF34FEEC222630772C201BF28D2FD8CA4F61CC643761C81C`。实现/Scientist/协议再次取哈希与§6/§1冻结值一致，回执source_unchanged=true，无结果后改门迹象。

### 7.1 数值来源与实际调用

P0记录A96实际类StateGatedMemoryROM、A16实际类MemoryTransportROM及对应MRO。两个模型的0.0025/0.00125s原buffer初始化追踪均经过initialize→encode→history_system，并各命中使用p_backward的源行10次。该计数是每次全队列批量调用的十段历史传播，不虚称逐事件产生了额外初始化调用；结果支持active_buffer而非dead_buffer。

两个模型具有同一唯一Q，数学核查已去重。80/120位矩阵参照差约1.02994e−81，120位解析与矩阵差约3.02546e−123，均满足登记1e−60尺度；固定人工零/常/线性资格通过。

相对高精度结果舍入至binary64的B_ref，旧buffer与当前Torch64均maxabs=1.5673018438633335e−10、相对Frobenius差6.6827409616e−11，数组哈希逐位相同；旧1e−12门仍fail，最大标准化违约76.5936387075。SciPy64相对B_ref差3.46944695195e−18，Torch32指纹差8.82712476447e−8。不得把前一个差值误称SciPy相对未舍入高精度真值的误差：公开记录SciPy对MP120的maxabs约1.06284510407e−16，B_ref自身舍入误差也为该量级，比较口径分别保存。

两模型四个被核JSON字段到加载binary64均逐位相同，当前证据不支持JSON加载精度损失；原g/rpp/S高精度装配相对主Q参照的差为8.67361737988e−19，不能解释1.5673e−10的主差。当前Torch64逐位复现支持同一算法在当前环境的数值差异解释；历史运行版本未重建，不能仅凭复现或dtype指纹宣布过去唯一原因、旧参数必然正确或曾使用float32。

### 7.2 因果初始化敏感性

四个模型×历史步长对照格均完整297事件，全部摘要available/denominator=297，递推、design逐位不变、QR传播、末点观测态不变及初始field关系五项检查均true。实际8次initialize和8次初始field，符合2模型×2buffer×2历史步长上限，没有为输出再次初始化。

| 模型 | 历史步长(s) | max绝对Δp_history | max绝对Δr0 | max绝对Δomegadot |
|---|---:|---:|---:|---:|
| A96 | 0.0025 | 4.60336979646e−9 | 1.30592514580e−9 | 8.57406234900e−9 |
| A96 | 0.00125 | 4.60336979646e−9 | 1.30592425762e−9 | 8.57406234900e−9 |
| A16 | 0.0025 | 4.60336979646e−9 | 1.30291466505e−9 | 8.61130189378e−9 |
| A16 | 0.00125 | 4.60336979646e−9 | 1.30291466505e−9 | 8.61130189378e−9 |

上述为登记坐标中的绝对变化，Δomegadot按rad/s²坐标理解，p及r不因表格数值自动获得SI功率/力含义。所有Δpdot均0，design差均0；历史p递推残差最大2.77830204750e−15，QR传播残差最大1.16073184185e−14，均过冻结atol=rtol=1e−12。公开结果同时保留逐事件max/median的归一化及相对变化摘要，未只选均值或把主细积分差混为buffer差。

这些证据支持误差产生与初值传递在当前登记范围可解释。变化幅度不能单独推出8s风险、关键事件分类或高阶物理稳定性无影响；本Run没有未来积分。父Run历史不相容297计数及反向块0/297仍为其原协议记录，不能由本probe的新参照和小幅Δr0自动消除。

### 7.3 执行与三个决定

公开回执记录唯一原生启动、退出0、源码不变、产物核验通过；原生wall6.7080261s，工具总wall5.9114941s。公开结果另列发表前wall5.7339353s，以及读取0.8685977s、参照0.2781326s、初始化2.4611096s。它们的起止范围不同，不能相加为总经过时间；人工资格、其他开销及产出在对应总范围内。此次真实启动无重试记录，§5仅为未执行候选的源码预检修订，不混成一次真实Run修复。

optimizer=0、共享参数估计=0、新ROM rollout=0、新FOM=0、future_truth_read=false、val/test=false、model_adopted=false。因果初始化的线性求解与QR敏感性核对属于原初始化，不是新的共享动力参数拟合。依据冻结实现的副本隔离与原state_dict前后检查、完成回执及公开结果，未见原模型或父Run被改写；本Reviewer未独立读取私有原模型再算其哈希，不夸称验证了全部私有资产。

当前建议代码retain_branch，模型retain_evidence/no_change，科学supported_within_scope限于三条已核数值证据链；历史唯一原因、物理机理和长期预测影响仍unknown。任何采用新buffer或改数值资格需要另行前瞻协议及原关键事件/原因果验证。本Run可进入结论稿与归档审查，不要求扩大确认或新训练。

## 8. 最终结论稿核对

本次追加前本报告SHA256为 `D23D30DF61B4E4C6C0CBE2F146A077E5F64C1345950BB0F12903EACCF9A2AD73`。实际读取结论稿 `reports/p_reverse_precision_20261031_conclusion.md`，初读SHA256为 `0EBB09652D459A2148110C8C9A2FBEB9C090F4E2574D9539121FEAC81F278B51`；公开probe及execution_receipt再次取哈希均与§7一致。未重算或读取私有载荷，旧p诊断报告未改。

结论正确区分B_ref舍入至binary64与未舍入MP120的误差，旧buffer/Torch64、SciPy64和Torch32的表值与§7公开证据一致。P0实际活跃、四格297事件、两模型各步长的Δp/Δr0/初始Δomegadot及递推/QR通过都保留其范围。没有把当前同算法复现写成历史唯一原因，也没有由小初值差声称长期预测或关键事件不变。

原生、完整工具及发布前wall三种范围分列，未相加，预检候选修订未被算作真实科学重跑。代码retain_branch、模型retain_evidence、科学continue分别表达实现保留、模型不采用与继续有界研究，与本审对三条数值证据链范围内支持的判断相容。父Run反向门失败及M3 unknown保留，后续最多两个数值候选仅为待采用方案，不是本Run已授权或执行的新预测。

唯一文字舍入修正：初读稿的“r0/rstar归一化L2变化最大4.07572e−9”应按公开最大 `4.075710921478693e−9` 写为 `4.07571e−9`。已反馈root；该修正不改变计算、阈值或科学判读，不要求重跑。除此未见结论范围或数值对应问题，待root保存这一末位修正后补记最终稿哈希即可归档。

## 9. 文字修正关闭与最终审查决定

**最终审查通过，无剩余归档阻断。** 本次追加前本报告SHA256为 `C4C8323F391752E05419826E3A20104CC9EFC15C6EB6DBAC0A0B792E3699AC83`。结论稿修正后实际SHA256为 `921B63DA32C28C128F08DC31D70F9C5C313F5A54A2662643A6CB4175B66E61C7`，已实际复读归一化L2句，最大值现为4.07571e−9，与公开值正确舍入一致；§8唯一文字问题关闭。

最终结论限于当前算法数值差可复现、原buffer活跃及100ms因果初值/瞬时导数差传递已核对；历史唯一原因、物理形式及长期预测影响不扩张。旧门、父M3 unknown、模型与数据保持原记录，后续方案未采用；代码retain_branch、模型retain_evidence、科学continue可归档。该末位修正只改报告文字，不涉及新计算、确认或源代码变更。
