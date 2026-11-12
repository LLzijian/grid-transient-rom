# 三点自治历史比较有限协议

状态：待独立审查／root采用；未登记、未释放。Run：`gftd_objective_alignment_20261112`；parent：`gftd_matched_prediction_20261109`。科学问题与解释范围见[设计](../../designs/gftd_objective_alignment_20261112.md)。只评价各自父到已保存唯一弱式提案方向的 t=1/2、1/4、1/8；两模型各一次方向，禁止新外循环。

## 1. 来源、身份和释放前置条件

沿用[父协议](../gftd_matched_prediction_20261109/protocol.md) SHA256 `0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43` 的§2–3逐候选求解、§6六门，以及其原尺度和合法输入合同；以下是本次明确变更，其他规则不变。父结果、原余额与暴露记录只读保留。本次没有端点重新评价或重建字典预算。

端点来源绑定 Coordinator 的 [endpoint_identity_digest.json](../../results/gftd_objective_alignment_20261112/endpoint_identity_digest.json)，SHA256 `ebc92ecdfbff3a182948569fc8ae3f60ed6dad458757bcafe5835824e23c07e6`；设计采用及后续准备回执同时绑定该SHA。摘要提供两模型父及唯一提案的实际functional、核、配置、掩码、未惩罚MSE／惩罚／目标、24包顺序、每包父best.u及原始残差hash、两起点、尺度、六门与运行依赖来源。旧独立逐包 `individual_packet_sha256=null`，不得补造；旧输入文件SHA和有序包聚合身份完整。摘要审查及下述新prepare身份核验未通过时，不具备科学执行条件。

聚合身份沿冻结源码 `gftd_matched_prediction.py` 原构造：`gftd_fitting._hash(dict(request=old_details.request, packets=ordered_validated_24, steps=core.HISTORY_STEPS, coefficients_sha256=old.FIXED_DP8_COEFFICIENTS_SHA256))`。这里request是每模型旧details保存的完整matched请求，不是新Run请求或inner.fit_request；序列化／转换规则必须与原源码相同。静态应精确得到 `663a1882ed49bc5680b6fe11ca9ceef2383f43ab02a5363226a2bfa480a49107`，调制应精确得到 `2f20dddccf309fec0eb45ca3b4ca22fc4c6c0ecce5a6e623c87912864be3660a`，均对应旧 `details.preparation_key`，不能混用inner的parent_preparation_key。原输入全字节SHA、slots顺序及两个内容聚合key共同建立旧有序24包身份；在其精确匹配且四个functional、配置、核心源和runtime均匹配时，允许复用旧端点评价，不需要科学重算。新逐包hash仅用于前向追踪，不成为声称旧单包hash存在的凭据。任一聚合key不符即停止，不重算端点、不改hash算法迁就结果。

提案如需从冻结 `raw_k/raw_beta` 重建，只允许用原 `_parameters` 作不求值的确定性构造，首先证明重建functional指纹与原实际被评估候选相同，随后才取有效c/beta。原掩码必须逐项等于实际 `c>0`、`beta>0`，对应惩罚必须与保存值一致。父和提案的非边字段必须一致；静态核 `exp_v1`、调制核 `bounded_exp_tanh_v1`；两端 `s_coi=0`，否则禁止沿用会写零的原构造器。任一条件不成立停止该设计，不用重算端点补凭据。

新编排器及纯mock测试可加入源清单，但每次评价调用的核心路径必须与旧端点精确对应：`gftd_matched_prediction._candidate`、原 `gftd_fitting`、`gftd_fit297`、物理functional、DP8及其传递依赖、SciPy/NumPy/Python运行版本和原记录的数值运行设置。冻结源字节和运行版本逐项核验；输入白名单、上述有序24包聚合身份、尺度、起点、sigma、盒、TRF、stepper、六门、残差和接受门均须一致。新增编排层不得改变调用语义；允许变化的只有本协议规定的内点边系数、对应模型hash／内容寻址身份和独立结果路径。整体新源清单SHA不同不是失败，但核心、依赖或语义不一致必须停止。运行依赖身份unknown不得靠重算端点解决。

root释放顺序：端点身份审查 → 本设计／协议独立审查与采用 → 隔离编排实现／源闭包／纯mock资格 → 新科学执行release与唯一job claim。本文不生成registry或release。每阶段保留receipt、输入／源／产物hash及真实计数。

## 2. 系数、实际回存与正支撑

从每个实际端点取 float64 `c=k_edge*s_edge, beta=c*s_mod, x=c-beta, y=beta`。要求全部有限且 `c>=beta>=0`，静态beta逐项为0。按固定运算顺序 `x_t=(1-t)*x0+t*x1`、`y_t=(1-t)*y0+t*y1`、`c_target=x_t+y_t`、`beta_target=y_t`，t精确取二进制的1/2、1/4、1/8。

复用原 `_parameters(parent,c_target,beta_target)`：正边s_edge=1，零边=0；k_edge=c_target；正c的s_mod=beta_target/c_target，c=0的s_mod=0。保存后重新读取实际 `c_actual=k_edge*s_edge`、`beta_actual=c_actual*s_mod`。接受构造需同时满足：

- 目标值和实际值全部有限；目标与实际分别满足 `0<=beta<=c`，门在[0,1]，静态实际beta精确为0；不clip、不补epsilon、不重阈值或去偏。
- 对c和beta逐项满足 `abs(actual-target)<=8*eps64*max(abs(target),tiny64)`，其中eps64为2^-52、tiny64为最小正规正float64；target=0时actual必须精确为0。目标正支撑必须等于实际正支撑；即便误差在容差内也不允许正数消失或产生新正数。
- 两类目标／实际正支撑分别等于各自两端支撑并集；检查核、所有非边functional字段（包括s_mem、s_coi）、delta_star和非模型身份配置不变。序列化回读后上述字段和实际有效系数必须精确一致。

保存两端与内点的目标／实际数组及hash、最大映射误差、两类支撑与计数。开放项数严格为实际 `count(c_actual>0)+count(beta_actual>0)`；惩罚 `1e-12*开放项数`。映射或存取失败属身份／构造失败，停止受影响模型，不改系数容差、数据或目标；另一完整身份模型可完成其固定三点，配对结果标不完整。

## 3. 六个新候选的逐次评价

固定24索引仍为 `0,12,24,37,49,61,74,86,99,111,123,136,148,160,173,185,198,210,222,235,247,259,272,284`，沿原event_id/local_slot和§1聚合身份。设计采用后的新prepare允许读取原297包合法历史JSON全字节，要求文件SHA精确等于 `b2607fc66c9bb96c5d6f85a6e074598e186ffab377c8aac3009699e1ea035817`；解析后只转换、验证固定24包，按§1分别重算两个旧details.preparation_key并保存比较回执，之后才可进入科学释放。核验不产生模型轨迹；其他事件不求值，不新增扫描或选择。此时新记逐包hash明确标为本次前向身份，旧值仍为null。严禁读取future缓存。p前10行不进入部署推断。k_ref、r_scale、a_scale=.1、历史归一化和物理常数均冻结。runner在调用前强验恰好24 packets和24 slots且顺序逐项一致，调用后强验24个事件记录，避免原helper非strict的zip静默截短。

传给 `_candidate` 的config必须是原fit297内层配置，不是fixed-DP8外壳。每个新候选照原路径调用 `gftd_fit297.make_config(new_f,mode,parent_inner.fit_request,297,parent_inner.preparation_key)`，保留全部非模型身份字段及原准备来源；新Run的编排／action身份另行绑定，不能通过改inner参数掩盖差异。端点摘要须证明旧提案也使用这一构造。任何确有必要的额外配置变更先停止并重新审查。

直接复用原 `_candidate` 及其共享helper，不调用旧 `fit_shared`、`_prepare`、`_audit`、`_diagnose`、NNLS或机制信息流程。每模型按t=1/2、1/4、1/8次序，候选始终由原固定两端产生，不由上一内点产生。模型之间可按资源并行，但同一结果路径单写者；每候选先claim，失败不得换ID重跑。

每包两固定起点：零；静态u_r逐节点交替±.1；调制u_a逐节点交替±.2、u_r=0。静态10维、调制20维；盒[-2,2]。各一次最多1200实际残差，TRF、x_scale=1、max_nfev=80、ftol/xtol/gtol=1e-10；所有差分、初始评价与内部回调计入。沿原eligible规则取两起点中的有限最小history_mse，并列取第一个。不做父u热启动或feasible注入。

历史采用float64固定DP8 .0025 s、.1 s逆向重建；同一完整ODE重新演化theta/omega/p/a/r，末点theta/omega/p₀按原硬锚。调制H_a包含beta*K'(h)*psi反作用，转速／p和转速／r双向端口保持原式；不得固定旧p/r/a轨迹。内点只改变边系数。

每个有24有限选点的候选逐包执行原六门：主／细历史往返全状态scaled max各<=1e-7；主细历史观测RMS<=1e-7；1 s自治主细观测RMS<=1e-5、peak<=1e-4；p₀误差=0。历史主/细 .0025/.00125 s、自治主/细 .01/.005 s，观测数值尺度沿父协议§6；这1 s是无真值的资格轨迹，不是L1。任一选点不合格则候选不能接受，不换次优点或重初始化；任一包无有限选点则按原helper将整候选记no_output，整候选六门均不跑，不补样本；W仍按24包尝试。

唯一选择目标 `F=mean(history_mse over 24)+penalty`；严格改善门 `new<old-1e-14*max(1,abs(old))`。每个内点先与各自固定父F比较，报告逐点资格和改善。最终从父及合格内点中按原严格改进门依次保留更好者，顺序父、1/2、1/4、1/8；门内并列保留先到者。t=1只作为旧已拒绝端点证据，不重新加入选择。三个点都评价，不因某点通过或失败追加／跳过其他点（身份／因果违规停止除外）。分别保存未惩罚MSE、惩罚、F和相对父的变化。

## 4. W的固定作用与两项比较

每候选每包独立一次W：取摘要绑定的父best.u，验证在原盒内；使用同候选配置的原 `gftd_fit297.endpoint` 构造端点，原 `_integrate` 固定DP8逆向.1 s，原 `history_residual` 于11点求原始未除优化器divisor的残差，`W=RᵀR/len(R)`。单次残差最多480 RHS；不通过优化器接口求W，不调用least_squares，也不调用 `infer_endpoint(feasible=...)`。W不使用新候选已选u。

保存 `W-parent_cached_history_mse`（同包父u、同尺度），以及 `W-candidate_selected_history_mse`（同一候选、同包、同尺度）。第二项为负说明返回值高于已知盒内残差见证，报告差的大小，不把极小舍入差称为已证实搜索不足；第一项不能单独定位候选优化不足。候选该包没有有限返回值时第二项为null。父缓存基准须由保存原始残差代数复算并与原history_mse核对，容差 `8*eps64*max(1,abs(cached))`，不符则停止复用，不能重新积分父模型。两种差的汇总分别只有24项全有限时给等权均值，否则对应主汇总null，有限项计数及逐包结果保留，不换分母冒充24包。

W不能预筛t、参与候选或端点选择、成为第三起点、改变父缓存、增加六门或触发额外求解。即使候选双起点／六门失败，W仍在该包固定预算内按计划尝试；失败写null。W无六门资格，所有改善表述均为残差信号，不称合格部署初态或参数—初态充分优化。

## 5. 纯代数／纯mock技术资格

科学源不改，只增加最小隔离编排与身份／映射校验。资格只用手写有限数组、字符串身份和mock回调；不读取原始事件，不调用真实functional RHS、H、梯度、ODE、优化、SVD、NNLS，不生成模型轨迹，不运行旧44项科学fixture或整套旧测试。构造／存取验证使用纯数据或不求值的fixture对象；涉及共享helper时先替换全部科学回调为mock，漏调立即报错。

固定六组检查各运行一次，最多两次针对已定位技术错误的修正后重查，共最多三轮；每轮最多1000次mock回调，合计3000次，无真实科学调用预算：

1. 三条手写边覆盖正c／零beta、c=beta、全零；两个端点支撑不同。三档t核对非负映射、并集、计数、门、核／非边字段和存取不变；另含NaN、beta>c、非法门、正支撑消失、容差内舍入与超差拒绝。
2. 手写端点／核心／依赖／输入／配置hash表，分别变更一个字段必须拒绝；只增加编排器身份允许，核心字节改变拒绝。两模型缓存和非连续global_index映射不串包；旧掩码不等于实际支撑拒绝。
3. 用mock两起点任务及六门返回，验证最小有限点、并列、到限但eligible、失败、缺一包、低F但六门失败；不得换次优点。分别模拟三个内点改进、门内相等与全拒绝，核对父比较及最终顺序。
4. mock共享helper验证两起点、1200上限、DP8、24包、每候选最多24六门、冻结尺度配置与实际正支撑传递；禁止调用旧fit_shared／prepare／audit／diagnose。用计数模拟确认差分回调计入1200，预算最后一次后再调用拒绝，不运行真实优化器。
5. 预设父残差、候选返回值与W，逐包复算两项差；W较好也不改变选点、初始化起点和候选顺序。非有限W令24包汇总null；W使用父u、同候选配置、原始残差尺度，不能混入优化器divisor。
6. mock异常验证claim/receipt、实际消耗在失败后仍保留、余额不转移、重复claim拒绝、禁止重算t=0/1、禁止真值路径和未来评分调用。构造RHS计数480/6480而不执行，验证科学总上限及分项不可挪用。

第三轮仍失败则资格不通过，停止执行准备，保留错误。任何真实RHS资格需求都不在本协议内，须先提出具体必要性、有限用量并重新审查；不得默许借资格启动科学计算。

## 6. 科学上限、异常和闭合

| 项目，两模型合计 | 上限 |
|---|---:|
| 新候选 | 6（各3点） |
| 优化任务／实际优化残差 | 288／345600 |
| 优化残差RHS | 165888000 |
| 选点六门 | 144套／933120 RHS |
| W | 144残差／69120 RHS |
| 科学RHS总计 | 166890240 |
| 新NNLS／SVD／独立H／真实未来／L1/L2/L3 | 0 |
| 纯mock资格 | 最多3轮、3000回调；真实科学调用0 |

独立H为0不表示RHS内部不求储能梯度；所有内部RHS按实际计数。每起点、每包、每候选余额独立，失败／早停余额不挪用，不转移旧Run余额。记录CPU、wall、线程、版本和失败成本，不新增进程数或小时限制。没有新端点复核轨迹、额外微分诊断、信息审计或随机种子。

身份、因果、核、输入或不可审计计数违规，停止受影响模型并保留账本；未知job先对账，不重领。普通求解／数值失败保留原24包，继续其余固定候选。科学执行后源需修改时先停止受影响阶段，由root审查技术原因；本协议不含受影响科学任务自动重跑额度。

所有三个点完成后结束该方向搜索。全无合格改善：保留父，停止更小t／更多起点／新轮；只支持这三个尺度和原算法下未获改善。内点合格改善：保留候选证据，支持原方向含可接受有限步，不宣称L1提升或机制恢复。W搜索不足或数值失败保留独立标签，不据此追加计算。三项决定分开，默认模型retain_evidence、default none；最终结论独立审查后由root采用。后续若要全量预测，必须另行设计、冻结和释放，本Run不自动延长。

本次只用原24包合法历史，无新FOM，val/test关闭，TASK4暂停；父train未来已消费，既不读取也不伪装成独立确认。释放前摘要、完整源清单、运行依赖身份、人工资格和root采用回执由内容寻址冻结清单补齐；只能补身份，不能用补清单改变本协议科学规则。
