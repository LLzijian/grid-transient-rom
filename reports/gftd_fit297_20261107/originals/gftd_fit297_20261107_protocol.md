# Run 3：train fit297 静态—调制同期比较协议

状态：Scientist最终设计，待独立复核、实现冻结及root释放；本文件不是启动凭据。采用唯一方案：全部合法fit297包做静态弱拟合，固定ID均匀最多24包做共享非线性适配，两方法分别对全部297包执行完整simple600→smooth1200及自治预测。输出支持本继承条件与有限预算下的同期观察，不称全部297均参与调制参数优化。

Scientist配置为gpt-6-astra，server_resolved_model=null。只读既有源码、静态元数据和冻结R_N参数；没有读取fit297事件、计算谱/证书、积分或拟合。报告使用anti-defensive-writing规则，事实和分母保持原定义。

## 1. 科学问题、父Run与释放顺序

问题是：同一合法信息、固定非边常数、完整初始化和共享适配机会下，开放边调制能否改变risk、共同omega、p和失稳检出？竞争解释包括短历史信息不足、耦合/隐态混淆、有限非线性优化、固定机械/记忆端口失配及24包激励覆盖不足。L2检验同一模型的局部吸引集，L3报告有效图与既有物理参考的对应，不成为L1门槛。

Run 1父提交为`3fd58f4313195d5f65ba1fa1a8931575b7638f45`。直接父Run 2仍在修复收尾：已冻结16个fit，38个完整初始化行持久化，随后写文件OSError 22；root正在审查一次无重放修复，身份相同的丢失输出拟复用已持久合法点，原丢失诊断unknown，剩余9个按原预算执行。这里不猜测父机制或预测通过。正式Run 3 release须引用父最终结果、修复回执、独立结果审查、root采用和归档commit/hash，核验无活动写者后才改Run 3来源并执行。当前允许完成设计和工程准备。

父代码因果/方程/数值合格但机制null时，本Run仍可作为用户已授权的真实条件探索；不以不断追加人工资格拖延主线。父实际代码错误须先修复。科学状态、执行状态与预算状态分开，COMPLETED不等于机制成立。优化充分性unknown不阻止本预算观察，也不支持机制否证或自动晋升。

原始val/test关闭，旧消费test退出选择，新FOM=0，TASK4暂停，default none，incumbent及受保护锚点不变。无新高阶条件、无新DMD研究、无付费资源。一次有界Run，无额外科学分支。

## 2. 冻结依据与采用理由

| 文件/资产 | SHA256 |
|---|---|
| `reports/gftd_fit297_20261107_design_review_preparation.md` | `0396a0549b27eccd4843c39b66f5d5fe4890811b18edafa77887edae92e59dbd` |
| `reports/gftd_fit297_20261107_performance_preparation.md` | `8476c49bd06531828ac5221254219256dd8eb2dca4f8524100999a8ecc742fd0` |
| `reports/gftd_fit297_20261107_l2_design.md` | `78dded5c0aef2a61339aa9e6c2383cf42385e53c1bc4691b8c0121c3c7b1fc4d` |
| `reports/gftd_fit297_20261107_l2_roundoff_review.md` | `699c67681488b9a74e56503f1ed199ee98cf0074634f423a1f474dc0ead63285` |
| `reports/gftd_fit297_20261107_kron_conventions_review.md` | `00b0200a222efeaB1ee433d3e4868c8244008c9abc27fb235f15df94a8a35545` |
| 冻结R_N export（第4节路径） | `937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9` |
| 原IEEE39工作簿 | `9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5` |
| `private_data/task2_fullorder_descriptor_0035/descriptor.npz` | `6648fca0340fbd0823b5f0d50e99f19ffa00ce6ee32bfb36002219938371c4da` |
| 同目录 `evidence.json` | `37d82fb8d1c3af8bb5d4cd7266c14d04913b10aa7caec77ac8beeb29336435fa` |
| `private_data/task2_verified_modes_0035b/evidence.json` | `8a1d3e53a1b4aeba5ceB2f1c7017378dc3ceB51be64a372a3562da8b9a7970ef` |

哈希比较忽略文本大小写，实际比原字节digest。root选择24包方案：N10两项最小等价RHS改动的固定数组计时约1.40倍，完整代价仍主要来自初始化；24包方案保留全体静态信息和最终评价，明显减少共享适配成本。不存在事后全量替代或两方案选优。准备选择记录为`results/gftd_fit297_20261107/preparation_authorization.json`；原工程bench.py/benchmark.json已原字节存于同目录`engineering_preparation/`，不重复执行已有量测。它们是静态准备和工程回执，不是事件读取/科学执行授权。

## 3. train-only读取、合法包、索引与缓存

使用`experiments/closure_probe_20261030.py::load_frozen_events`绑定的`checked_preparation(...,fit_only=True)`路径，验证prepare/cache/source/schedule哈希和完整fit ID，不加载no_update、val或test。不使用configs/dataset.json旧canonical入口。原分母固定297=267稳定+30失稳；event ID与标签留在runner，仅用于身份、原范围统计和最终评分。

每事件五键为`theta_history,omega_history,history_times,p0,sigma`：theta=x[:11,:10]+delta_star，omega=x[:11,10:20]，p0=x[10,20:30]逐值复制，时间为[-.1,0]的11等距点；sigma=0，表示按当前记录硬锚端点，不表示记录或FOM无误差。禁止额外键、p历史、未来Pm、真隐态、故障信息和事件类别。initialize只收上述前四键，sigma/尺度来自存档。原始x的定义/单位与登记不符时typed停止，不靠猜测换算。

按ID转为原登记字符串后逐Unicode码点字典序排序（不按数值重解释、不依赖locale），检查唯一性及与prepare的身份集合相等。合法性失败保留ID和原因，不补替事件、不以标签挑选。记合法数n；n=0 typed FitFailure且模型不更新。m=min(24,n)，共享适配索引为int(floor(j*n/m)),j=0..m−1，验证唯一且0<=index<n。读取后、任何优化前持久冻结全部合法排序ID、非法ID和索引映射；算法本身在本协议已冻结。n<297时仍保留原分母，并明确不是297个可用输入均拟合。

`fit(train,val,run_context)`接收全部n个合法五键包，val=[]。run_context的专用`gftd_fit297_v1`配置只含适配整数索引、算法/尺度/来源hash，不携带标签/未来；完整ID映射存runner私有receipt。模型内部验证索引并构建一次全体preparation，两方法使用同一哈希缓存；子集引用逐事件缓存，不重新构造p/r_f或重跑全体静态筛选。公开通用fit接口仍拒绝未知配置/非法包，副本完成并验证后原子更新模型；FitResult.best_validation_loss=null。

两格每包一次p/r_f缓存；键包含有序五键输入、全部固定常数、网格和尺度配置。全体弱矩阵按sqrt(40n)缩放，适配按sqrt(40m)缩放；不得将全体归一化矩阵直接截行当子集矩阵。参数变化必须重新推断a和自治历史，p/r_f可复用；任何常数、输入、顺序或尺度变化使对应缓存失效。save/load保存preparation来源hash、适配索引/数量、fit诊断，原观测与大矩阵留私有侧，不以路径指针静默换输入。

## 4. 同一方程、常数来源、单位和继承暴露

来源为`private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json`。只读JSON有效state_dict直接buffer，不加载/运行父模型；κ=κ_reference exp(κ_raw)、λ=λ_reference exp(λ_raw)。提取完整数组、原字节hash、源格式/代码hash归档；缺失、符号/维度错误typed停止，不补全1默认值。

| 新模型字段 | 冻结来源/值与含义 |
|---|---|
| M、delta_star | R_N mass、delta_star；相角rad、omega rad/s，沿原M3坐标，不把机器Sn上的ANDES惯量直接代入 |
| D、G、R_p、S_p | damping、jpi_p、rpp、storage_base；既有已学闭合常数 |
| eta | K10字典序Bᵀdelta_star的实际存储binary64；L2保留其舍入值 |
| 评价/历史尺度 | p_scale、input_scale_delta、input_scale_omega、rstar，分别按原定义使用 |
| 记忆 | J=κI，K_r=lambda_mem=1/κ，tau_mem_sync=1/(κλ)，R_m=κλI，s_mem=1 |
| 幅值 | K_a=lambda_amp=I，gamma=.5逐节点，tau_amp=5即R_a=.2I；明确固定建模先验 |
| 门 | 状态耗散门、memory Laplacian关闭，s_coi=0；tau_mem_coi显式1但不起作用；不偷偷恢复共同omega解耦 |

理想参数化给出动量记忆力`−r`与`rdot=κomega−λr`；lambda_mem不是衰减λ。实际存档Kr=float(1/κ)、tau=float(1/(κλ))后，连续实数ROM严格使用`−J*Kr*r`、`rdot=J*omega−(Kr/tau)*r`，按各存档因子作精确实数运算，不反推或强称乘积恰为1、衰减恰为理想λ。新模型共同采用对角Q=λI，取代R_N原空间记忆图；不是R_N精确嵌入，不调用仅接收RMemoryROM的from_parent，也不继承其预测成绩。机械端口始终`pdot=−G omega−R_p S_p p`，未来无Pm强迫。

状态为[theta,omega,p,a,r]共50维，储能

`H=.5(omegaᵀMomega+pᵀS_p p+aᵀK_a a+rᵀK_r r)+Σ_e[K_e−beta_e+beta_e exp(gamma_i a_i+gamma_j a_j)]psi(phi_e;eta_e)`，
`psi=cos eta−cos phi−sin eta*(phi−eta)`。

实际相角力与幅值反作用均取同一H偏导。H为ROM储能，不称SI焦耳；gamma*a无量纲，a/r不指认器件状态。只在报告中做COI投影，自治状态保留完整omega/p/r。

已查量级：M .1250738–3.1804463，D 4.99247e-8–.2336076，κ≈.8562738254，λ≈.2668975413；delta输入尺度.200618–.686860 rad、omega尺度.969175–2.280011 rad/s、rstar .225630–.530801。26个正参考边权的标准中位数k_ref为排序第13/14项均值≈1.7374097018；实现从冻结数组计算并保存binary64，不用打印舍入值替代。

逐字段继承声明：M、delta_star、D/G/R_p/S_p、κ/λ、全部尺度、rstar、p_scale和k_ref均来自旧已选择R_N/M3谱系，继承其既有train和旧test暴露。delta_star同时用于相角恢复、eta和L2/L3中心，不称独立物理真值。k_ref只传入旧边权量级，不传入26边支撑，但仍是已有选择信息。原source_identity的M3 SHA为`c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5`。本轮train-only不抹掉历史暴露，不构成独立泛化确认。

## 5. 新配置分派、两个方法与有限共享拟合

新增显式初始化配置格式`gftd_causal_two_stage_fit297_v1`及对应fit配置，不改Run 2的`gftd_causal_two_stage_v1`定义/全局协议常量。旧legacy_static、旧Run 2模型load保持原算法和严格validate_config；新格式缺字段/模型指纹不符typed拒绝，不回退encoder或补新默认值。

新config完整保存：本协议hash、物理/有效K/beta指纹、模式、sigma、Q构造版本`mgs2_p0_ordered_positive_v1`及Q、逐机theta/omega/r尺度、a尺度.1、盒/起点、k_ref/门值、网格、目标、差分/优化器参数及调用上限。求解器、`_endpoint`、`history_residual`、后向筛选和initialize都从该新格式读取同一配置；旧分支不读新字段。变量r=rstar*u_r，u_r∈[-2,2]；a=.1*u_a，u_a∈[-2,2]，rstar只改变搜索坐标。sigma=0删除theta/omega修正变量，端点硬锚；新格式本Run拒绝非零sigma，避免引入未登记的噪声自由度。

新历史残差为 `R_j=[P(theta_hat−theta)/input_scale_delta,(omega_hat−omega)/input_scale_omega]`，P=I−1Mᵀ/sum(M)，按通道逐元素除尺度；使用完整omega误差，不投掉共同omega。共享目标 `L=mean_j mean(R_j²)+1e-12*(edge_mask_count+mod_mask_count)`，j仅24适配包。局部残差除1e-8只作数值调理，记录原尺度SSE与优化尺度分别的梯度/KKT。弱式仍force/.1、投影后除sqrt(total_rows)，是另一个条件目标，不能混作L。

static：beta=0、a_end=0；全体静态筛选信息/求解/一次去偏合格才用其k，否则全45边k=k_ref、beta=0先验，保留null和原数值未决原因。modulated：全45边k=k_ref、beta=.25*k_ref探索。两个方法非边常数、24索引、外循环、初始化预算相同。静态beta=0标fixed_by_model，不算发现无调制。

每方法初始候选+3提案，每候选每事件两个起点，各200次实际历史残差调用。第一起点全0；第二静态u_r=.1*(-1)^i，第二调制u_a=.2*(-1)^i、u_r=0（i零基）；端点无修正。当前候选a历史生成精确指数列，`u=k−beta,v=beta>=0`与[A_static,A_exp]做受限NNLS；记忆干扰逐事件40×10块SVD消元，不构造巨型全局记忆SVD。

每次后向至多90轮，每轮比较所有可删父边/可关闭调制，固定45边顺序，同边先删父边。仅删除、不重加；当前解缓存。必要比较任一NNLS非有限/失败/相对KKT未过使整提案unresolved，拒绝，不删失败项后选优。NNLS最多200迭代；相对KKT审计沿冻结Run 2/Run 1源码。罚项按开放掩码计，不按瞬时参数阈值重计。

后向结束只应用一次k>=.05*k_ref、beta>=.025*k_ref且父k过门，再一次无罚去偏。去偏跨回阈值则raw参数/掩码保留、相应support=null，不重复筛。共享提案用新参数重新推断24包，再以L严格改善 `Lnew<Lold−1e-14*max(1,abs(Lold))` 接受；同分保留旧候选。拒绝消耗一轮并恢复incumbent参数/掩码/合法隐态，不从拒绝点延续，不按未来选参数。无合法初始候选typed FitFailure，父模型不变。

局部使用SciPy trf、max_nfev=80、ftol/xtol/gtol=1e-10、无量纲变量Jac绝对步2.5e-5，边界二阶单侧；所有Jac/初值/守卫评价计实际调用。合法零边、无改善、budget_limited分别记录；固定条件NNLS数值最优不等于联合非线性充分。模型全部选定后存档hash，未来评分不回写选择。

## 6. 信息判断、全体补充分析与有限分解数

机制解释列为[k,beta]与[A_static,A_exp−A_static]，不同于NNLS的[u,v]列。主/细两格共享矩阵报告绝对RMS列范数、奇异值、1e-8/1e-10/1e-12相对秩、每列投影掉其他列后的条件比例/秩稳定及相关性。沿父tau_abs=1e-6、条件比>1e-8、相关性<1−1e-8、两格秩一致；关门不自动删除完整未知列的信息要求。信息失败使相应恢复support=null，raw模型仍能合法预测；全模型联合可辨性保留unknown。

N10每事件40弱行，记忆消元最多10维，秩最多40且典型最多30，不能独立识别45静态/90解释列。逐事件只做一次解释矩阵奇异值分解和绝对列范数，报告维度不足/秩，不做必然失败的逐列投影，也不要求逐事件全列合格。恢复资格以共享批次判断；逐事件结果不是总资格门。共享每矩阵做1次全SVD及每列1次其他列SVD，即static46、modulated91次；使用同分解同时算三阈值，不重复。

最终24包适配条件信息固定两格各一次。全部合法完整初始化结束后，另对每模型实际成功返回点的a历史作一次全体覆盖的两格条件信息；只使用各自成功ID，不用失败点a=0补齐。两范围的ID/数量/目标分开，该全体信息不回选或再拟合。返回点历史可复用同点缓存，缺失时至多每配对主/细重积一次。

信息SVD上限：初始全体静态46；24范围两格共享274、逐事件最多96；部署范围两格共享274、逐事件最多1188；合计1878次解释/信息SVD。逐列条件投影数=(45+90)*2*2+45=585，每事件条件投影数0。逐事件记忆Z块的SVD另设4000次保守上限，覆盖全体准备、6提案、重推后检查和两范围信息投影，缓存减少actual；不得按提案每候选重复完整信息审计。保存planned/actual/skipped与复用来源，超上限停止该分析为unknown，不扩预算。无K10枚举、profile扫描或补充seed。

## 7. 完整初始化、同点数值与工作量

两方法各对全部合法包从头完整simple600→smooth1200，不复用训练代理作部署起点。simple从r=0，仅估r，末点theta/omega/p硬锚、a=0；非零beta下历史a仍按完整ODE演化。simple合法输出自身资格通过后smooth从其r、a=0开始；static估r，modulated估a/r，预算1200。smooth非改善返回simple守卫点并记录no_improvement；stage1失败阻断stage2，缺输出/数值未决typed失败。budget_limited但自身资格通过可返回，优化充分性unknown。无细格再优化、无额外随机起点、无旧encoder回退。

历史主/细.0025/.00125 s，未来.01/.005 s，float64。数值资格沿父原尺度，不随新训练尺度改变：全50维往返差/.1最大绝对值<=1e-7；11点[Ptheta,omega]向量主细差/.1 RMS<=1e-7；p末点逐值误差0。初始化含同末态1 s未来比较RMS<=1e-5、peak<=1e-4。原实现若同点已检查可复用hash回执，不能用别的点认证最终点。正式8/15 s另按原真实系统门逐时域评估，不把1 s门当长时资格。

| 工作项 | n=297,m=24 上限；失败及早停均保留分母 |
|---|---:|
| 全体两格p/r_f准备 | 2n=594重建，两个方法共用 |
| 初始静态+6提案 | 7×(90×91+1)=57,337代数项，NNLS不超过此数 |
| 代理 | 16m=384局部任务、3200m=76,800残差积分 |
| 完整部署 | 2n=594配对、4n=1,188局部任务、3600n=1,069,200残差积分 |
| 合计优化 | 1,572任务、1,146,000残差积分，45,840,000历史RK4步 |
| 未来 | 每配对粗1500+细3000步，合计<=2,673,000 RK4步 |
| 完整初始化点资格 | 每配对最多2点，每点历史往返240步+1 s未来300步，共<=641,520步 |
| 24适配最终信息细历史 | 两模型24点×80步=3,840步 |
| 全体点信息补历史 | 每配对最多120步，共<=71,280步 |
| 信息/记忆分解 | 信息SVD<=1878、共享逐列投影<=585、记忆块SVD<=4000 |
| L2 | 13半径/模型，共26；点包络<=594次；无epsilon扫描 |
| 平衡及模态 | 两模型各<=20 Newton更新、每更新<=10线搜索；各2个49维Jacobian和2次普通谱；FOM原始+行均衡共2次699维QZ；Kron一次29维求解及一次奇异谱 |

登记积分合计上限49,229,640 RK4步（非线性、未来、点资格和两类信息重积相加）；端口重建用另一组RHS单独记，不藏在此数。Newton最多每模型20个19维中心差分Jacobian，各38次梯度评价，另最多200线搜索与21中心检查，合计<=1000次梯度评价/模型；不借Newton做模型参数优化。两部分优化预算不转移，失败任务不能以重启补满另一类。

仅采用性能准备中两项改动：幅值梯度向量化为`gamma*(abs(B)@(active_k*s_mod*exp(h)*psi))`；两个状态耗散门关闭时直接返回常数Ra/Rm。保留shape/finite检查、门开启旧分支，不再顺手删校验或加依赖。父Run 2归档后在独立Run 3快照实施，做梯度/功率、非法输入、旧load/静态兼容和主细数值检查。数学同式不承诺浮点按位相同。

性能准备固定N10 RHS外推24方案历史残差RHS项约3.87小时、全部登记积分RHS项约4.16小时，不含优化器、代数、I/O等；不是完成时间承诺。记录实际每阶段wall/CPU、计数、线程/设备、失败成本。最多两轮明确技术修复，不改变科学输入、门、预算，不重置已用机会。未知job先对账，模型/事件完成后小型原子receipt，避免巨大partial反复覆盖；丢失诊断unknown，不猜数或重复科学动作。

## 8. L1：冻结评分、原分母和数值分辨范围

参数和全部初始化结果冻结后，评分器才读取同一train事件的未来。预测从每个实际返回的同一个50维末态分别粗/细积分，不再次initialize；不调用旧40维占位或会固定Torch模型的旧evaluate_model，也不运行旧107条val evaluator或自动更细重跑。保存实际50维轨迹；指标适配为[theta−delta_star,omega,p]30维。未来Pm只能是最终评分真值的一部分，绝不进入ODE。

复用`_point_error`、`risk`、`event_metrics`、`physical_metrics`、`_pair_row`、`scientific_qualification`、`interval15/actual_interval_predictions`的原定义并冻结源码hash。准确的NumPy适配、资格门、LOS及输入字段映射绑定`reports/gftd_fit297_20261107_l1_adapter_contract.md`（SHA256 `2d09f180266875a6100c15c61cc915fc5a2e7752d1de641dce6be603d6c74252`）；带Torch或固定8秒的旧函数按其中公式最小转写，不直接调用高层评价器。risk为逐事件时间平均归一化点平方误差的.8事件均值+.2最差ceil(10%)均值，不是RMSE。1/2/5/8 s按fine[::2][1:100h+1]对x[11:11+100h]；粗同步索引。各完整时域稳定原分母267，失稳真实区间原分母30，n_i=min(1500,len(x_i)−11)，不足时间不补尾段、不补零。

分别报告共同omega的惯量加权共同分量RMSE（rad/s）、p/p_scale归一化RMSE、稳定risk各时域、失稳真实区间risk及相应时域/点数。共同omega/p的池化权重沿原代码，逐事件值也保存。8 s分类与真实区间<=15 s分类独立给漏检/30、误报/267、未判定/未覆盖及ID；检出时机T仅在共同可比事件集合配对，不将8 s更多检出解释为实际记录内更早。

`_pair_row`的LOS资格不替代误差数值资格：每个risk、共同omega、p时域均应用上述原函数的主细门及分数扰动统计，冻结适配的准确函数/字段调用映射；实现审查必须核出该映射，不能只给一个passed布尔值。不得降低原门。主细两分数差、每模型数值未决ID以及配对差的扰动之和同时报告。

任何无输出/数值未决/未覆盖均保留297/267/30及594原分母。全范围risk有缺项则null，已覆盖描述统计和双方共同子集只作带ID的辅助结果，不替代主分母。静态减调制的差只有超过双方离散扰动范围才称本格可分辨，仍不自动等于机制识别。关键错误身份并列，均值不抵消失稳漏检变化。历史R_N、21/30只作明确模型/原范围reference，不计本轮配对差、不重新训练。

成本仅在同设备、精度、步长、事件、时域和批量条件下比较，分别报初始化、积分与总耗时。没有同期FOM成本则加速“未比较”。L1效应、优化健康、充分性、机制信息和证据有效性分别判断。

## 9. L2：同一5N模型的严格舍入残差证书

使用已审原推导及舍入补充，不要求eta浮点精确循环一致，不改任何模型参数。认证对象为每个存档binary64参数因子取其精确实数值后定义的连续ODE。令F=Fraction.from_float，严格使用K=F(k)*F(s_edge)、beta=K*F(s_mod)，其中s_mod为实际存档并经合法性处理后的门值；不能先用binary64连乘得到舍入积，再将该积转Fraction而声称认证同一H。gate/tau、存档J*Kr等实际算子也按因子精确计算；若参数已由倒数或乘积构造后存入binary64，则以该已存值为准，不反推理想κ/λ替代。报告可列舍入后的K/beta浮点值，但严格证书保留有理乘积。无需新增beta物理字段或改变原方程，存取逐位核验。普通浮点特征值正数不作为证明。

中心theta_c为存档delta_star减精确有理均值，其余omega/p/a/r为0。只商去常数全局角，保留完整共同omega。令phi_c=Bᵀtheta_c、d_e=|phi_c,e−eta_e|、Gamma_e=|gamma_i|+|gamma_j|，A0=max_active max(|phi_c|,|eta|)。球范数为相角1正交分量与全部omega/p/a/r的固定数值坐标欧氏范数，不混入历史归一化尺度，不称物理距离。用`sum(dtheta²)−sum(dtheta)²/N`精确计算相角范数，无需把浮点Q当严格正交。

先精确核0<=beta<=K、活跃图连通、M/S_p/K_a/K_r>0、D/R_p/R_a/R_m严格正；缺一则not_certified及原因。lambda_mem=1/κ是储能，不能替换λ。取Gamma_max=max_active Gamma，Rcap=min(1,(7/5−A0)/3,1/(2Gamma_max))，Gamma_max=0忽略最后项；Rcap<=0停止该模型证书。固定R_j=Rcap/2^j,j=0..12。

每R计算alpha=A0+1.5R<7/5，h=R Gamma_max<1，cos_lower=1−alpha²/2>0。因sqrt(2)<1.5且1.4<π/2，共同cos正域保证psi>=0。连通图lambda2>=ell=4Kmin/[N(N−1)]，取

`mtheta=cos_lower*(1−h)*ell`，
`C=Σ beta_e Gamma_e*(2R+1.5d_e)/(1−h)`，`ma=min(K_a)`，
`det=mtheta*ma−C²>0`，`mu=min(det/(mtheta+ma),min(M),min(S_p),min(K_r))`。

这些全为有理保守界：exp(−h)>=1−h、exp(h)<=1/(1−h)；混合块两个sqrt(2)给2R，中心偏差给1.5d_e。mu是整个49维球的Hessian正下界。中心梯度上界 `g=1.5ΣK_e d_e+.5Σbeta_e Gamma_e d_e²`，且0<=H_c<=.5ΣK_e d_e²。仅严格g<mu R/4继续，rho=mu R²/4。13候选按rho最大、同分小R选择，不看事件/标签/覆盖。

强凸给球边H>=H_c−gR+.5muR²>H_c，故存在唯一内部极小点x*；||x*−c||<=g/mu，H_c−g²/(2mu)<=H*<=H_c。实际平衡omega=p=r=0，theta/a可有舍入偏移。V=H−H*的球内V<=rho集合非空、凸、紧致且不碰球边；能量不增保证正向不变。完整p/r交叉功率抵消，严格耗散使零耗散最大不变集只含x*，LaSalle给到平衡相移轨道的距离趋零。无需声称全局相角α收敛，无FOM安全结论。

点能量使用固定有限有理Taylor包络：sin取Σ_{k=0..18}(-1)^k x^(2k+1)/(2k+1)!，余项包络±|x|^38/38!；cos取Σ_{k=0..18}(-1)^k x^(2k)/(2k)!，余项包络±|x|^37/37!（Lagrange导数绝对值<=1）。exp取0..18阶，|x|<1时余项包络±|x|^19/[19!(1−|x|)]。有符号区间运算，有限阶无法分开门则unknown，不无限加阶。Hc_lower=0合法；`V_upper=H(z)_upper+g²/(2mu)`，要求点严格在球内且V_upper<rho，报告`margin_lower=1−V_upper/rho`、R/rho/mu与包络。

每个实际合格ROM返回点最多一次包络，共<=594；未初始化点未覆盖。没有独立确定性epsilon，真实初始化集合覆盖unknown，sigma=0/小历史残差/success不提供epsilon。本Run不执行epsilon扫描或假设epsilon=0。若报告公式说明其所需条件，Haa上界必须含舍入d：`D_R=max(K_a)+Σbeta Gamma²*(d_e+1.5R)²/[2(1−h)]`，与`A_R=2ΣK/(1−h)`及C形成上界；不得直接继承精确中心的R²项。证书失败不推断模型/事件失稳，rho不称临界能量/切除时间。

## 10. L3：唯一端电压Kron与既有descriptor物理模态

有效图报告K、beta、raw/null支撑和参考状态角度刚度；图是有效相角作用，不等于线路图。L3所有结果report-only，无最少匹配数门。

### 10.1 已采纳bus-terminal参考

唯一版本为`bus_terminal_line_shunt_pqz_casev0_s100_v1`，实现逐项沿已冻结Kron审查。耐久工作簿`private_data/gftd_fit297_20261107/reference/ieee39_full.xlsx`，retrieval收据`results/gftd_fit297_20261107/case_retrieval.json`；官方blob `f49523f01b15d833c4376f020f4ba4846478e8e3`，原临时下载路径只作溯源。zip/XML读取，系统基值100 MVA/60 Hz，各bus电压基值Bus.Vn。Line设备以bus1基值换算，阻抗乘Zn/Zb、导纳乘Zb/Zn，本case系数均1；不重复加Vn1/Vn2或2πf，不按GENROU Sn换Line/PQ。

Line含12变压器全部采用：ys=u/(r+jx)、yh=u(g+jb)/2、y1=u(g1+jb1)、y2=u(g2+jb2)、m=tap exp(jphi)；Yff+=(ys+y1+yh)/tap²，Yft+=−ys/conj(m)，Ytf+=−ys/m，Ytt+=ys+y2+yh。tap在bus1侧、phi为rad；Shunt对角+=u(g+jb)。不加epsilon，不取q绝对值。

19个PQ包括保留端负荷，固定阻抗`Yload=u*(p0−j q0)/Bus.v0²`，p/q已是系统基准；Bus.v0只作声明锚点，不称验证潮流。此PQZ是本参考假设，不冒充历史FOM实际ZIP，亦不加PV/Slack/GENROU源内阻。保留R=bus30..39、消去E=1..29，`Yred=YRR−YRE solve(YEE,YER)`，不用逆/伪逆或正则修补。

一次YEE奇异谱记录条件数，线性求解和Schur关系归一化残差均<=1e-10，复对称转置误差<=1e-12（phi=0），否则unknown；不要求Hermitian或零行和。归一化分母用max(1,相应矩阵Frobenius范数)，记录定义。只报告45边|Yred_ij|与ROM有效耦合的排序关联、分别归一化权重和边作用分布，零方差排序相关为null；不报告物理真边FP/FN、不强行同单位。内部EMF Kron本Run不做。

### 10.2 冻结FOM模态与实际输出形状

参考仅用已接受699维原始DAE descriptor；拒绝原160维linearization.npz的6正模式。原source evidence遗留160/6是历史字段，不覆盖已审150有限、549无限、1转动零、0正实部结论。物理签名`e52c4d30ec9db0ae76c6a2f1f722ca00a7a5803c4dbb01ad3b6718bfcade16d9`、ANDES2.0.0、case hash、已保存全/代数平衡均核对；state_names前10 delta GENROU1..10、后10 omega同序。与fit297登记物理签名/机组顺序/冻结平衡不一致则FOM匹配unknown，不靠case名继续。不运行ANDES，不重导出矩阵。

对既有pencil作一次带右/左向量的广义eig/QZ，沿原`audit_private_modes.py`有限规则：|beta|>1e-10*(|alpha|+|beta|)；0/0为|alpha|+|beta|<1e-10，必须0个。另一次按row_scale=max(rowmax(|A|+|E|),1e-10)行均衡QZ仅作谱核查，有限数一致、原一对一谱差<1e-6；原有限对相对残差<1e-8。数目/参考极点与已接受证据不符则unknown，不挑删异常极点。

参考选择为原接受的全部13个正虚部>1e-6且GENROU angle/speed参与度>=.1、参与度可识别模式，保留9个.641–1.687 Hz及4个.0606–.1828 Hz，不能按新模型匹配好坏缩成9个。参与度沿finite descriptor projector `v*w^H*E/(w^H*E*v)`，|overlap|<1e-12者不伪造参与度。已接受13个身份与重新计算数值以极点匹配核验，不以top_states代替完整振型。

FOM右向量取实际10个omega行，ANDES pu转canonical rad/s乘冻结omega_s=2π*60（核元数据一致），再乘diag(sqrt(M))。ROM相同diag(sqrt(M))作用于完整10omega；两边不删除共同omega，各自复向量L2归一化后MAC，统一单位/权重记录。全局复相位不影响MAC。

### 10.3 ROM平衡、差分和全配对

从theta_c,a=0起，在theta的9维Q坐标与10维a求完整H的驻点，omega=p=r固定0；最多20次Newton、每次10个步长1,1/2,…,1/512，按梯度范数严格下降接受，当前点已过门即停止。Newton Jacobian中心差分h_j=1e-6*max(1,|y_j|)，奇异/非有限/无下降停止unknown，不加ridge或改变参数。若有L2获证球，所有接受点留球内；无证书则只接受总相角/a位移范数<=1的候选，平衡求解仍不证明吸引性。

完整RHS平衡残差maxabs<=1e-10才作正式平衡模态；未过时可报告原中心残差但模式匹配为unknown，不把非平衡Jacobian称机电模态。对合格50维平衡构造T=diag(Q,I10,I10,I10,I10)，只删共同theta的49维RHS，固定Q并核QᵀQ/1ᵀQ误差<=1e-12（这里是数值模态，不用于严格L2）。中心差分两个步长h_j=1e-6*max(1,|x_j|)与其一半，比较49维Jacobian相对Frobenius差/max(1,||Jfine||F)<1e-6；每格98 RHS调用，共每模型196。用细格Jacobian，各格一次普通eig以检查极点；相对特征对残差<1e-8。任一资格失败该模型匹配unknown，不细步追加。

ROM候选固定正虚部>1e-10、频率闭区间[.01,10] Hz、极点模>float64 eps，物理10omega输出有限且范数>eps，沿`state_space_modes`归一化；所有排除项和原因保存，不按阻尼或匹配误差删模式。两格全部候选用极点距离Hungarian对应，候选数或边界归属不同则模式数值未决；配对极点相对差|lambda_main−lambda_fine|/max(1,|lambda_fine|)<1e-6。未决不通过更改候选频段修好。

正式13个FOM参考与全部合格ROM候选按现有`match_dmd_modes`全配对Hungarian，显式`max_relative_frequency_error=inf`，cost=相对频差+1−MAC，继承`evaluate_event_modal_matches`合同而非helper默认.5。给所有分配的频率、阻尼、MAC、差值及全部未匹配项，不事后剔除高误差，不设置MAC成绩门。两个模型共用同一FOM右向量及物理参考，不生成新DMD/事件时窗。

## 11. 实现范围、审查出口与收尾

最小改动覆盖：train-only runner及五键/四键适配、显式新配置分派与逐机尺度、全体准备/24索引复用、两项RHS等价改动、有限信息审计、L2有理helper、已有模态与Kron的本Run调用。调用方/存取/typed原子失败/缓存和计数测试同步；不建新执行平台、不改旧模型/评价定义。

确定性工程检查覆盖非平凡新逻辑：合法切片/索引无标签、不同总行数归一化、尺度变量映射、r动力学映射、旧格式回归、新格式完整存取/指纹、主细同点与50维占位、异常保留分母、有限预算、L2包络及舍入非精确中心、Kron方向/负q符号、49维保留共同omega。复用现有fixture/stub，不新增科学样本或训练。Run 2冻结源在其归档前保持不动。

冻结清单包含本协议及采用回执、父Run 2闭合与归档、全部源/环境/hash、常数/尺度提取和暴露、数据身份与输入规则、预算、精确评分调用映射、Kron/descriptor来源、Scientist与Reviewer身份。仅形式check通过不代替root释放。独立Reviewer逐项核本协议第3–10节及实现，root作一次精确释放。

实际结果留五轴：证据有效性、优化健康、优化充分性、机制可检验性、效应与确认范围。闭合分别给代码、模型、科学决定；本协议不预设晋升。L2点获证不是FOM安全，L3对应不是线路恢复，单次fit297匹配不叫独立确认。失败及阴性结果入证据档案，原分母与预算关闭记录不删。

私有观测/轨迹/checkpoint/大矩阵/优化细节留合法私有目录；公开材料限协议、汇总、来源hash及允许的证据。按research_policy选择性提交、普通推送、原字节核验，模型指针独立。至此完成设计，不授权本文件生成时执行。
