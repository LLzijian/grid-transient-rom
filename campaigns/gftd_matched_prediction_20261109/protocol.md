# 匹配共享适配、原297预测与同一方程分析协议

**本 Run 预先固定静态／有界调制两模型的适配、完整初始化、L1 分母和 L2/L3 分析，以一次有限主线比较回答真实预测及其解释。**

Run：`gftd_matched_prediction_20261109`；类型：adaptation + matched comparison。状态：`CONDITIONAL_PREREGISTRATION_PENDING_CURRENT_CLOSURE_AND_QUALIFICATION`。架构：[设计](../../designs/gftd_matched_prediction_20261109.md)。协议在当前 full272 结果尚未闭合时登记，不依赖其未见结果；当前 Run 的额外工作授权为 0。源设计、独立数学／公平性审查及 root 采纳见设计首节，实际 configured_model=`gpt-6-astra`、server_resolved_model=null 的回执保留。

## 1. 父模型、合法信息与释放顺序

| 固定来源 | 身份 |
|---|---|
| 静态父模型 | `artifacts/gftd_initialization_precision_20261109/candidate/pilot/static_model.json`；SHA `ed79e2cdea1f6830c052b9dae8fb8c2b748170e8e15ec0a5a8e3034d72b787cc` |
| 有界父模型 | `artifacts/gftd_bounded_initialization_20261109/candidate_model.json`；SHA `f8eabacfeac00074f135c7af826d8d89d8f23c445c788485d0fb79a0f54b0637`；kernel=`bounded_exp_tanh_v1` |
| 297 合法包 | `private_data/gftd_fit297_20261107/inputs.json`；SHA `b2607fc66c9bb96c5d6f85a6e074598e186ffab377c8aac3009699e1ea035817` |
| 原事件顺序 | SHA `9a76d08d0423302a2b07374e9f2a92646db3be089cadeebaadace5b985699de4`；train297=stable267+unstable30 |
| 当前 Run 完整闭合、科学判读、独立审查 | pending/null；由后续只追加 freeze receipt 绑定实际路径／SHA，不改本比较 |

模型身份来自当前公开 prepare 元数据和旧 pilot execution，不读取新观测推断。两父模型的既有全297选择暴露保留；当前有界父 refit=false，不能称新核已公平拟合。所有非边物理常数及原 input_scale_delta/input_scale_omega/rstar/k_ref 从冻结父请求核对，双方应一致；不一致先对账，不另估尺度。原 exp 仅作旧证据锚点，不增第三模型训练。

只允许原100 ms 11点 theta/omega、相对时间、末点真实 p0、sigma=0及冻结常数进入 fit/initialize。ID/标签只作索引／报告，故障身份或未来不可进入方程。原 val/test 关闭，新 FOM=0，TASK4 暂停。真实 train 未来只在第6节冻结之后由独立评分器读取；本准备阶段原始输入和未来读取均为0。

释放阶段固定为：A 当前 Run 闭合+独立审查+26文件原字节归档；B 本设计／协议独立审查、实现源闭包及人工资格；C 两模型共享适配／历史诊断；D 冻结两模型后完整初始化／参考／自治轨迹及 L2/L3；E 冻结全部初始化选择、评分器源及真实未来来源后评分。每阶段先保存 root release 和唯一 job claim，结束保存 receipt／数组哈希／计数。A 之前只可在新路径准备隔离源和 mock 测试，不编辑或执行当前26文件；这些新文件的准备另由 root 安排。技术门不是新的用户确认请求。

## 2. 公开接口与数值合同

新增 `RunContext.gftd_matched_prediction_v1` 显式请求；它与旧 `gftd_fit297_v1` 同时非空时拒绝。请求精确绑定 run_id、本协议 SHA、父模型 SHA／kernel／mode、297计数、24索引、原尺度／k_ref、源闭包和缓存身份。未登记字段或未知版本拒绝。`GFTDKuramotoModel.fit(train, [], context)` 仍执行合法五字段白名单与 sigma 检查，调用新的继续适配模块，返回标准 FitResult；不经旧 fit_shared 重置参数。无新训练后门。

输出沿已有 `make_fixed_dp8_config` 及有效内层 fit297 初始化配置，保留 600/1200、r_scale、a_scale=.1、[-2,2] 与历史归一化。内层兼容的旧 proxy_start_calls=200 不用于本轮共享适配；新 RunContext 和 fit_state 明确记录 1200、2 starts、3 proposals、完整库及新协议，不能沿旧配置字段宣称本轮预算200。新 preparation key 为本轮内容寻址身份，旧 key 仅来源。公开 initialize、save/load、clone 继续验证同核和 model_sha；默认 exp 的旧指纹／载荷不变，bounded v3 拒绝降级。新模块不能靠 monkeypatch 永久改旧默认算法。

所有自治代理残差、细历史审计、部署初始化及预测显式使用 float64 固定 DP8；历史主/细 .0025/.00125 s，未来主/细 .01/.005 s。系数 SHA `123669c4e6be7b9e45302adb0f7714a9972c8f1e196728914b43b2bd9c2f6361`。观测驱动 p/r 端口重建保持原独立时变 RK4 双网格与弱式求积，回调另计。

## 3. 两模型的有限共享适配

固定原均匀24索引 `floor(j*297/24)`：

```text
0,12,24,37,49,61,74,86,99,111,123,136,
148,160,173,185,198,210,222,235,247,259,272,284
```

释放清单绑定这些 global_index 的原 event_id 与 local_slot=0..23；只准备两模型各24包两格缓存。不得按全297索引直接访问24槽，不能让单槽 `_CACHE.clear()`丢掉另一模型。真实数据内容不进入公开日志。

各自父 functional 直接作首 incumbent，有效系数 c=s_edge*k、beta=c*s_mod；未评价前不重置或把一般 s_edge 静默二值化。每模型初始代理加至多3个共享提案。每次提案重新开放 K10 全45边：静态只开静态列；有界45条静态及调制列都可选。原有限删除/关闭、阈值 .05*k_ref 与 .025*k_ref、一次去偏、NNLS maxiter=200/相对KKT<=1e-9不变。必要 NNLS 比较失败令该提案未决；不绕过失败项后选优。拒绝提案不替换 incumbent。

每候选每包两个固定起点：零；静态交替 u_r=±.1，有界交替 u_a=±.2 且 u_r=0。首节点取正，后续交替。无旧最优隐态热启动。每起点单次1200实际残差上限，所有中心／单侧差分计入；TRF、x_scale=1、max_nfev=80、ftol/xtol/gtol=1e-10不变，早停照常。无条件重启、第三起点或续窗。

两起点选有限最小历史MSE；共享目标为24包等权历史MSE加1e-12实际开放项数，接受需 `new<old-1e-14*max(1,abs(old))`。并列保留先来的 incumbent／起点。任一包无有限代理则该候选目标无效。

每个具有24个有限选点的候选，在这些唯一已选端点逐包执行第6节原 DP8 六门，全部通过才有候选数值资格；初始候选也执行。最多2模型×4候选×24=192套，新增1244160 RHS。原1 s未来门仅为自治数值一致性，无真实未来。资格失败令提案不能替换incumbent；首候选无效或失格令该模型fit_failed，保留父文件但不在本次主比较补位。无替换起点、次优点回退、参数调整或门槛放宽。另一合法模型继续。每次资格保留原端点身份、各门和计数，异常／未执行明确记录；不假定捕获可复用以缩减最终细历史审计预算。此门在本条件预登记中前瞻新增并由root采纳，源草案保持原哈希。

新核仍用 K 独立于门的字典 `(A_static,A_K)` 和系数 `(c-beta,beta)`；条件解释坐标为 `[A_static,A_K-A_static]`。完整库重开不等于真实线路识别。

## 4. 最终代理的固定充分性与条件信息检查

每模型只对最终选中24个 best.u 检查一次。使用保存的原尺度残差 R（220维）及主历史，另在同一点用 h=2.5e-5、h/2=1.25e-5 求可行残差 Jacobian：盒内中心差分；靠边按原二阶单侧公式，中心残差每步长最多求一次。每个10/20维点两步最多42/82次残差。任何非有限列令该点诊断未决，不换步或剔列。

各步 `g=J^T R` 是半平方误差梯度，非优化器 `g_opt`；sigma=0 时 `g_obs=1e-16*g_opt`，MSE梯度=2g/220。投影沿原1e-7边界容差：下界且g>=0、上界且g<=0的分量置0。固定相对量 `eta=||Pg||inf/max(||J||F*||R||2,1e-16)`；步长一致量 `d_g=||Pg_h-Pg_h/2||inf/max(||Pg_h||inf,||Pg_h/2||inf,1e-16)`。记录列范数、两个 eta、d_g 和原终止／活动边界。局部一致门 d_g<=.1；较细 eta<=1e-3 只标记“小相对投影梯度”，不称充分优化或联合可辨识。

若细 Pg 精确全0，方向为零，两个方向点跳过且计数0；否则 d=-Pg/||Pg||inf，各一次 `clip(u+.01*d,-2,2)`、`clip(u+.05*d,-2,2)`。与原点精确相同的试算跳过；非有限保留失败。报告SSE、差和相对降低 `(SSE0-SSEtrial)/max(SSE0,1e-16)`；超过1e-6标记有可行改善证据。诊断不回填best、共享选择或部署初态。到限／无同点梯度／不一致／仍有改善分别记录；无这些标记也不是全局最优证明。两模型合计最多3072额外残差。

最后各24包主细条件信息审计一次，主历史复用、细历史48条DP8；按原去记忆投影、三阈值秩、列范数、条件比例／相关性报告。共享审计覆盖全部45/90解释列，不以当前gate删列；逐包仅谱和范数。上限信息SVD370、条件投影270、缓存记忆SVD96。调制条件信息不足时 support=null，joint_identifiability=unknown；不阻断合法L1，不新增全297信息扫描。

## 5. 人工资格：固定输入、无ODE或真实优化

下面全部人工检查与至多两次针对明确缺陷的复查共用128 RHS、2000 H；直接梯度／核／字典、有理包络和mock回调分别计数。不得整套运行带 synthetic_candidate 的旧测试。无随机fixture、人工ODE、人工初始化或真实fit优化。

物理fixture精确继承[储能协议§3](../gftd_storage_feasibility_20261109/protocol.md)：N=3完全图；M=(1,1.5,2)，k=(1,.8,1.2)，theta_star=(.1,-.05,0)，eta=B' theta_star，gamma=(.5,-.25,.75)，储能／端口／时间常数／记忆连接均1，D=.2，s_mem=1、s_coi=0，无状态耗散门。状态i为theta_star其余0；ii为theta=(8,0,0)、omega=(.1,-.1,.05)、p=(.2,0,-.1)、a=(.1,-.2,.3)、r=(.01,-.02,.03)；iii/iv仅将ii的a替换(±2000,0,0)。A门edge=(1,1,1)/mod=(0,0,0)，B=(1,1,0)/(.5,0,0)，C=(1,1,1)/(.5,0,1)。i/ii各A/B/C、iii/iv仅C，共8候选点；i/ii的A另作2个原exp静态对照。仅6正常候选点做全H梯度差分，步1e-5/5e-6、atol1e-7/rtol1e-6；8点有限／功率门1e-11*max(1,|power|,dissipation)，静态atol1e-12/rtol1e-11。饱和点不做H差分。ii/C额外用于一次save/load、deepcopy和规范变换身份检查，调用均计入总额。

字典仅预设历史 t∈[-.1,0] 的两原网格、theta(t)=theta_star+t*(.2,-.1,.05)、omega=(.2,-.1,.05)、a=0或(t+.1)*(1,-.5,.25)，p/r手工缓存全0；A/B/C门检查K列独立、 `(c-beta,beta)`弱积分组合，atol1e-11/rtol1e-10，不重建端口或积分模型。

新增纯mock固定为：①24个槽位值等于global_index，event_id=`fixture_`加三位索引；核标签static/bounded分别保存，测试不连续索引与双缓存不串包。②三边手工掩码(1,0,1)变成完整候选，三个预设提案目标[.9,1.1,.8]与初值1，应保留第1/3、拒绝第2；另测试完全相等、缺一包、NNLS失败，以及目标改善但一包六门失败时拒绝且不换次优点；初始候选一包失败则fit_failed。③public fit上下文互斥／未知字段拒绝、参数复制、合法五字段、DP8 residual/候选六门/细历史/initialize/rollout dispatch及预算1200；优化器/积分器mock，只记录stepper，不调用模型。④存取旧exp省略字段、bounded v3拒降级／未知核、fit_state和initializer合同区分；保存前后哈希核对。⑤预设零数组(101,50)/(1501,50)/(3001,50)，单一已知列误差、NaN、缺行、主细LOS单侧触发与初始化失败，核验原297分母、official=null、单位及禁止截短；无生成轨迹。

诊断mock固定u=(0,0)、(-2,2)，R=(u0,u1,1)，解析J；另R常数(0,0,1)给零梯度；模拟单一非有限列验证未决。只求手工多项式，不求模型；覆盖中心／单侧两步、投影、相对分母与两个方向点，测试记录不冒充科学残差。

L2纯有理fixture固定：h∈{-1/4,0,1/4}，量化数{0,±1/3,±2^-129}，检查整数floor/ceil尤其负下端点；Taylor19项的正负输入区间、E下界正性、外层指数域(-1,1)、区间顺序及18次有符号区间累加。输入E区间[-1,1]、反序区间和外层端点1均应拒绝。半径fixture A0=0、G_K=G_beta=1给cap=1/2：j=0因b过滤、j=1通过；beta空则b=0且不得调用核包络。c=(1,1,0)的三节点连通和c=(1,0,0)的不连通、beta=(1/2,0,0)及beta全0、delta=(0,0,0)与(1/1024,0,0)、严格正耗散与一个零耗散分别走固定分支。边集精确按>0，不设剪枝阈值。数学资格是解析不等式和有理区间保序，不能用几个模型数值点取代证明。

## 6. 完整初始化、参考和真实L1

两最终模型及选择轨迹冻结后，各按原297顺序完整初始化一次：simple零起点/a=0、最多600残差；stage1六门通过后smooth从原映射起点、最多1200残差。原最小同一步长历史损失选择不变，stage2获选失格不回退；theta/omega/p0硬锚、a=.1和rstar尺度、[-2,2]、TRF及容差保持当前初始化协议。每包最多两套六门，保存各腿数组及端点hash，主细轨迹不得由重新initialize取得。

六门沿当前协议：主／细历史往返全状态scaled max各<=1e-7，主细历史观测RMS<=1e-7；1s主细未来观测RMS<=1e-5/peak<=1e-4；p0误差=0。数值观测为质量质心去相角、完整omega，再除.1，区别于拟合原尺度。六门数组按端点／方法匹配保存；记录包装try/finally恢复，worker内部无并发initialize。

每个正式六门通过点最多一次Radau 1s：rtol1e-11、atol1e-13、max_step .005，共同输出.01、真实RHS<=24000含数值Jacobian。以已存细DP8为主确认：整段完成、观测RMS<=1e-6、peak<=1e-5、全状态相对max `|x-y|/(1+max(|x|,|y|))<=1e-5`；主DP8同门另列。不一致不重试，也不把该较严门偷偷替换原六门或原长时合同。

每个正式返回点各一次15s DP8 .01/.005，主(1501,50)、细(3001,50)，包含全部内部子步；不重复初始化，不插值补首秒细步。参考失败保留独立标签，长时资格按原合同自身判断。共享模型与所有初始化选择冻结后评分器方可读取原train真实未来；不因预测、L2/L3或评分修改参数、选择点或追加fit。

L1完整沿[原NumPy适配合同](../../reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_l1_adapter_contract.md)，SHA `2d09f180266875a6100c15c61cc915fc5a2e7752d1de641dce6be603d6c74252`；其明确的对齐、质量投影、尺度、p_scale、LOS及eligible规则是本协议组成部分。主项稳定267完整5s risk，另报1/2/8s；失稳30按各真实可用区间<=15s。细[::2]正式评分，主同范围单列；物理共同omega RMSE(rad/s)、归一化p RMSE、8s/实际区间分类和触发时机分别报告。

任一eligible失败则该范围official=null；初始化失败保留原ID，不转为unavailable删除。共同成功子集仅辅助配对结果。主细全部50维有限、COI及共同omega归一化RMS<=.001；LOS-aware原双触发<=.02s、单侧触发不合格、延迟未覆盖等均保留。预测不超过实际真值长度评分，不把15s自治输出当成15s真值。评分只做一次既定遍历，不按结果改分组或指标。

## 7. 最终ODE的L2严格局部证书

新模块显式分派已知kernel，拒绝未知核；静态保留原证明。用有效c>0精确边集E_K及beta>0精确子集E_beta，图连通／c_min／相角域用E_K，核导数与混合项用E_beta，不删小正边。浮点中心与eta转精确有理，theta用精确均值去公共角；delta=|phi_c-eta|，A0=max_EK(max(|phi_c|,|eta|))，Gamma=|gamma_i|+|gamma_j|。

固定 `G_K=max_EK Gamma`，`cap=min(1,(7/5-A0)/3,1/(2G_K))`（G_K=0略最后项）。cap<=0直接未获证。候选 `R_j=cap/2^j,j=0..12`；有界另过滤 `b=R_j*max_Ebeta Gamma<=1/4`，E_beta空则b=0且不求核。不得预缩cap后重新生成13半径。各半径仍检查alpha=A0+1.5R<7/5、全部正性、连通和严格耗散；通过者选最大rho，同分选较小R。

K=exp(tanh h)，K'=K(1-t²)，K''=K(1-t²)(1-2t-t²)。b<=1/4给 `K>=1-b, K'<=1/(1-b), 0<=K''<=(1+2b)/(1-b)`，且psi>=0。固定界：

```text
ell=4*c_min/[N(N-1)]
mt=(1-alpha²/2)*(1-b)*ell; ma=min(lambda_amp)
C=sum_Ebeta beta*Gamma*(2R+1.5delta)/(1-b)
det=mt*ma-C²
mu=min(det/(mt+ma),min(mass),min(storage_p),min(lambda_mem))
g=1.5*sum_EK c*delta+.5*sum_Ebeta beta*Gamma*delta²
rho=mu*R²/4; require det>0, mu>0, g<mu*R/4
A=2*sum_EK c/(1-b)
D=max(lambda_amp)+(1+2b)/(2*(1-b))*sum_Ebeta beta*Gamma²*(delta+1.5R)²
L=max(A+C,D+C,max(mass),max(storage_p),max(lambda_mem))
```

舍入中心c不是精确平衡。强凸保证球内唯一极小点z*，距中心<=g/mu<R/4，球面相对能量>9muR²/32>rho；H(c)>=0给 `H(z*)>=-g²/(2mu)`。点条件为未归一化5N商空间范数严格<R且 `H_upper+g²/(2mu)<rho`；omega/p/a/r全保留，theta仅去共同均值。所有damping、rate_p为正且幅值／记忆耗散在球上一致正，才由LaSalle得趋向真实局部平衡相位轨道；零耗散返回未获证。未知epsilon保持null，不拿残差代替确定性初态误差球。

真实核能量用原19项有理Taylor：包络exp(2h)为正区间E，通过严格递增(E-1)/(E+1)得t区间，再包络两个端点exp；外层exp前、每完整函数包络后和能量累加后固定2^-128向外量化，整数floor下界/ceil上界，严禁浮点tanh或向零截断。检查指数域(-1,1)、正分母、区间顺序；量化造成边际不获证就记录，不加精度／阶数。每E_K边至多3次三角包络，E_beta边额外3次指数包络。

每模型13半径、每个正式合格返回点一次包络，最多26半径/594点/160380科学Taylor调用。没有新增浮点H/RHS、epsilon扫描或覆盖搜索。未获证不等于物理失稳，证书只针对最终ROM。

## 8. L3与实际平衡

每最终模型一次原有限平衡／谱流程：最多20步19维Newton，每步最多10次回溯；原收敛及失败合同不放宽。只删除共同theta，49维Jacobian按1e-6与5e-7各一次中心差分/eig；原模态稳定性、输出可观性、[.01,10]Hz频段、Hungarian匹配及误差阈值保持。精确来源为冻结 `experiments/gftd_fit297_analysis_20261107.py` SHA `2faa8cdd83a98f71c044e20983b2b08d0f394f76b64c15b24f2c63617dab929e`，新模块复制经审的纯分析逻辑，不import旧runner。

复用已冻结13个FOM机电模式与Kron缓存，源路径和hash在释放清单绑定，未绑前不执行L3；不新增FOM/QZ/DMD。若无合格平衡或差分不一致则保留未决，不补种子。报告实际a*、平衡残差、全部匹配／未匹配及频率、阻尼、MAC；零幅值二阶一致不用于解释非零或不同平衡处的谱差。无最低匹配数前置门，也不凭谱对应宣称非线性调制被识别。

## 9. 有限调用上限、异常和闭合

| 工作（两模型合计） | 硬上限 |
|---|---:|
| 共享代理 | 384优化任务，460800实际残差，221184000 ROM RHS |
| 每候选24个选点的原六门 | 最多192套，1244160 ROM RHS；无新优化 |
| 最终48点充分性诊断 | 3072残差，1474560 ROM RHS；0新优化 |
| 6次完整库提案 | 49146代数项／NNLS上限，每NNLS maxiter200 |
| 两核24包两网格准备 | 96记忆SVD，46080强迫端口向量回调，非ROM RHS |
| 代理条件信息细历史 | 46080 ROM RHS，信息SVD370，条件投影270 |
| 完整初始化594包 | 1188优化任务，1069200残差；513216000残差RHS+7698240六门RHS=520914240 |
| 严格Radau1s最多594条 | 14256000 ROM RHS |
| 15s主细最多594对 | 32076000 ROM RHS |
| L2 | 26半径，594点，160380科学Taylor包络；0模型RHS/H |
| L3 | 434 ROM RHS，1960独立梯度，4 eig |
| 科学合计 | 1572优化任务，791195474 ROM RHS |
| 人工资格及至多两次技术复查 | 128 RHS、2000 H；无ODE／优化 |
| 全RHS合计 | 791195602 |

一次历史残差40 DP8步×12=480 RHS；一套六门540步×12=6480 RHS、每包最多两套。所有差分／求解器内部回调包括数值Jacobian计入真实RHS，不能用nfev代替。直接梯度、内部梯度、H、mock、端口、SVD、Taylor分别计数，不相互冒充。各任务／包余额不转移，不因换ID、后处理失败或重prepare重算。技术修复最多两次且共用人工上限；科学开始后源变更先停止受影响阶段，保存失败与已用账本，再由root按同一协议审定，不重置额度。

遇身份／因果／核违规停止受影响执行；普通数值失败保留分母继续其余固定任务；未知Job先对账。无有限首代理的模型后续标not_run/null，不借未来结果补位。条件信息、L2证书数量、L3匹配数及代理必须胜过旧模型均不是科学执行门。当前Run若最终不能闭合审查，本Run维持未释放，不能绕过前置证据。

完整记录wall/CPU、线程、残差与各类真实调用、失败成本；不恢复进程数／小时限额。所有job先claim后receipt，source closure含公共源、新模块、runner、评分器、测试及库版本；旧26源归档引用与最终新源分别保存。模型／配置／输入／协议／源／数组均有哈希，原始观测、隐态和轨迹留私有artifact，公开results保留汇总、身份及成本。收尾保存原分母行表、L1/L2/L3、五轴判断、独立审查及代码／模型／科学三决定；未执行写null，模型默认retain_evidence、default none不变。

冻结前尚缺的仅是当前Run闭合产物、实际新实现源、24 ID映射与真实未来／参考缓存的具体来源清单。它们由append-only freeze receipt补齐，不能改变已登记对象、门槛、预算、选择或失败合同；任何需要改变这些科学规则的发现须先形成明确修订并重新审查，不悄悄改协议后宣称预登记。
