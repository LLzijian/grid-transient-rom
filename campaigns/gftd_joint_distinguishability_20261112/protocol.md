# quarter固定调制方向补偿见证：有限协议草案

状态DRAFT，未登记／未释放。Run为 `gftd_joint_distinguishability_20261112`，parent为 `gftd_objective_alignment_20261112`。只检验[设计](../../designs/gftd_joint_distinguishability_20261112.md)的局部存在见证，不作不可补偿或全局辨识的结论。

## 1. 冻结对象与合法信息

参考functional、inner config、24个已选u和对应尺度全部来自parent的 `modulated_quarter`。job SHA256为 `7b98ad2b68220d6599baaaa71dced108d574af643bece9ae8bc45ab2e9c5b35a`；其系数／明细hash由job精确绑定，实际选点以保存best.u为准，不重新初始化。beta_parent取同一parent所引用的旧调制父实际回存系数，方向唯一为 `d_beta=beta_quarter-beta_parent`。来源资格沿parent [result_review](../../reports/gftd_objective_alignment_20261112/result_review.md)及audit SHA `93a455515512885196cc993c5fcb2758f255768df46b99f11d4178522b291ebb`。

只继承父prepare已验证的24包缓存，不重开原297包。父 `results/gftd_objective_alignment_20261112/prepare.json` SHA为 `995399315f5d32d8e7f50d3894f7879f1a810e98cae73695581923f5043b7cef`，绑定 `artifacts/gftd_objective_alignment_20261112/prepare/packets.json` SHA `1cc6cb649cc9aaa073a5eb996cf279097fe7a968e678f09a4a057edb20aba403` 和同目录 `modulated.json` SHA `8c478b04b9dfb005bbfc24d7649aa8108f32482c5e1f92faea91bd8be8a93f51`；前者提供合法包，后者提供已绑定的旧调制父系数。逐项核对父prepare的slots和前向包hash。继承父审查已验证的两个旧details.preparation_key：静态 `663a1882ed49bc5680b6fe11ca9ceef2383f43ab02a5363226a2bfa480a49107`，调制 `2f20dddccf309fec0eb45ca3b4ca22fc4c6c0ecce5a6e623c87912864be3660a`；旧独立逐包hash仍为null，父prepare前向hash保持原口径。缓存缺失或身份不符即停止，不重开原数据补救，不读取未来、其他样本或新FOM。

quarter job绑定的 `artifacts/gftd_objective_alignment_20261112/science/modulated_quarter/coefficients.json` SHA为 `19e16fd0d7eeb457e10ffe46bde384dcce34bc1354d93d0d9e91016ef527d992`，同目录 `details.json.gz` SHA为 `07faaa42dd44e0b0dfa245f18ac079cf1de5649f6f176254d48bd56892c0b0aa`。前者提供实际functional及inner_config，后者提供24个best.u和原始残差；按job、父prepare的slots和审查绑定逐项验证。所有科学调用前绑定这些来源、内核源码和runtime版本；原45源档案SHA `10560d308499f0f15f5dac20f520db262ffd9cf77be813aa29a2083daadec3c9`是来源，新编排独立冻结。

原 `gftd_fitting._parameters/_integrate`、`gftd_fit297.make_config/endpoint/history_residual`、DP8及物理核复用原字节。禁止调用fit_shared、infer_endpoint、initialize、数值六门或评分器。新的配置只按原make_config更换实际模型身份，其他尺度、物理常数和preparation来源不变。源／数据／配置身份不符停止，不重算旧候选补凭据。

root采用与独立设计审查在先，随后实现／纯mock资格、源与输入冻结、独立执行release。本文不提供release；旧工作余额已关闭且不转移。

## 2. 变量、有限集合和实际映射

参考为 `x*=c*-beta*`、`beta*`、`u*j`。k_ref取quarter系数缓存 `inner_config.fit_request.k_ref`，须有限为正且等于冻结公开值 `1.737409701777866`；幅值、记忆及节点尺度取同一inner_config，不另算尺度。p₀与末点theta/omega固定；u前10维为幅值、后10维为记忆，实际 `a=.1*u_a`、`r=r_scale*u_r`。bounded核与全部非边参数固定。

取 `alpha_max=1/64`，两档alpha为 `{1/64,1/128}`，只正号。对照为 `x=x*, beta=beta*+alpha*d_beta, u=u*`。补偿为

```text
x=x*+alpha*k_ref*v_x,  beta=beta*+alpha*d_beta,
c=x+beta,  u_j=u*j+alpha*v_j。
```

因此总delta c=alpha*(d_beta+k_ref*v_x)，不能把对照叫纯beta-only。所有45条共享x可变化；v_x与24×20个v_j堆叠为525维。固定可行区间：

```text
lower_x=max(-1, -x*/(alpha_max*k_ref)), upper_x=1;
lower_uj=max(-1, (-2-u*j)/alpha_max),
upper_uj=min(1, (2-u*j)/alpha_max)。
```

全区间须有限、包含0且lower<upper，否则停止方向不可测试，不删列修补。|v|<=1是人为局部补偿范围；x>=0和u物理盒是原模型约束，二者分别报告。beta*+alpha*d_beta在两档必须非负；d_beta全0则方向为空，结束无新科学求值。不能clip beta、u或候选系数以挽救不可行点。

用原 `_parameters` 回存c/beta，按parent协议§2的8eps64相对映射容差、零值精确及所有非边字段不变检查；存取回读实际值一致。正支撑一致只比较**本次扰动target到actual**：二者的 `c>0` 与 `beta>0` 掩码各自精确一致，不要求与quarter参考支撑相同，参考x=0允许向正方向进入。每次差分／完整核验均使用实际回存functional，保存目标与实际系数。中心和单侧均使用原有符号扰动，每次直接核验beta>=0、实际c>=beta及u∈[-2,2]，不只检查线性方向，不靠clip或临时变更模板救回。对照、补偿或solver解越界时，该条件无有效观测，保存状态与已耗工作，停止该条件核验并记未决。

## 3. 完整观测映射、差分与数值门

每包定义R_j为原 `history_residual` 返回的220维**未除优化器divisor**向量，包RMS定义为 `||R_j||₂/sqrt(220)`。先以原endpoint构造状态，再 `_integrate(...,backward=True,dense=True,duration=.1,stepper=dop853_step)`，主网格.0025 s共40步、细网格.00125 s共80步；主格密集结果每4行、细格每8行取样，均为11点，经同一原尺度残差函数。每映射分别480／960 RHS。完整p/r/a随自治ODE演化，无观测驱动端口替换。比较向量差R(扰动)−R(参考)，不比较历史标量损失。

每网格每包首先算一次参考R。主格参考与保存quarter raw residual最大绝对差须<=`1e-12*max(1,maxabs(saved))`；不符停止，不重新选择u。全部参数和输出均float64且有限。对主／细每个对应映射，逐包RMS差须<=1e-8；这里只是历史一致门，不是原六门或未来资格。

每包66个方向：45个x方向（单位变化为k_ref）、20个本包u方向（单位1）、一个固定beta方向d_beta（固定x，故c同时变化）。差分步固定h=2.5e-5和h/2=1.25e-5。每方向每步两个扰动求值；参考R复用。以较大h先决定模板，两档和两网格使用同一模板：两侧均实际可行则中心差分；否则选能容纳2h的可行正向或负向（二者均可时正向），用二阶单侧 `sign*(-3R0+4R(sign*h)-R(sign*2h))/(2h)`。没有可行模板则整项未决，不改h、删列或换符号重试。beta方向的实际回存同样检查可行性。

每列都核验：(a)细网格两h差；(b)较小h主细差。逐包列向量RMS差须<=`1e-6+.05*max(RMS(column1),RMS(column2))`；任何一列失败则线性化未决，停止矩阵求解和有限核验，不重差分。该门结合双h／双网格检验可复现性，不是导数真误差的严格上界。

另定义每个原220维R的舍入指标 `e(R)=64*eps64*max(1,maxabs(R))`，R已按原节点尺度但未作堆叠归一化。步长s的中心列指标为 `(e(R(+s))+e(R(-s)))/(2s)`；二阶单侧为 `(3e(R0)+4e(R(sign*s))+e(R(sign*2s)))/(2s)`，包括单侧R0。比较两列C1、C2时，取二者列指标之和；该和若大于相应一致门右侧 `1e-6+.05*max(RMS(C1),RMS(C2))` 的1/10，该比较未决。双h和主细两项都按此判门。该指标是分量误差包RMS的保守筛查，不声称覆盖ODE内全部舍入传播；实际回存误差、缓存一致性、双h和双网格各自保留，不能互相替代。

全部门通过时，只用**细网格较小h**组装A、b。每包的45共享列填入同一列块，其20局部列填入自身块，其他局部列为0；b取beta方向列。按固定包序将24×220行堆叠，并除sqrt(5280)，所以欧氏范数是全体归一化观测RMS。A形状5280×525；没有正则化、先验新权重、截列或SVD筛选。A、b、B、D及线性预测一旦如此归一化，后续所有向量差、门右侧和eta均直接用欧氏范数，绝不再次除sqrt(5280)；只有单包220维门使用包RMS。

## 4. 唯一线性补偿问题和两档核验

解一次 `min_{lower<=v<=upper} .5*||A v+b||²`。固定SciPy原runtime的 `lsq_linear(method='trf',lsq_solver='lsmr',tol=1e-10,lsmr_tol=1e-12,max_iter=100,lsmr_maxiter=1000)`，不热启动、不续窗、不重启。外围计数限制：1次有界求解、最多101次内部LSMR（含初始无约束调用），每次最多1000迭代。已审本地源码表明初始LSMR至多一次、TRF每次外层迭代至多一次；守卫必须同时覆盖 `lsq_linear` 与 `trf_linear` 模块各自绑定的 `lsmr` 别名，入场前查调用余额并固定maxiter，按返回的实际itn记账。外层保存status、nit、message、cost、optimality和可行性；内部逐次记录入口、序号、maxiter、istop、itn及残差诊断，异常也记已发生调用。出现额外矩阵求解即停止。无NNLS、SVD、QR／直接分解求解或非线性least_squares。

冻结SciPy 1.18的三个本地源（相对site-packages），新prepare和执行前须核对，不导入另一份同名实现：

| 源文件 | SHA256 |
|---|---|
| `scipy/optimize/_lsq/lsq_linear.py` | `9a6e3a1e3f965510ec49c193d86543f897e6d7a89f2962aec4eb9e0b73af3b9d` |
| `scipy/optimize/_lsq/trf_linear.py` | `184306c8ca96c1a54b51aeec0a8f62c0399492e4fbe3aa1c8b588ae8a8e72500` |
| `scipy/sparse/linalg/_isolve/lsmr.py` | `73d4b3c402e20dd08e8e7690184253c4b755dc75c2b98f76361dee798596bb9a` |

保留返回状态、实际迭代、残差和可行性。任何有限可行v即使到限也可进入固定核验，因为见证不依赖最优性；无有限可行v则结束未决。绝不将求解器目标当下界。v不替换保存quarter u，不成为新部署初始化器或模型采用候选。

同一个v在两档alpha、两个网格上各求24包的两种完整历史：对照(x/u不变)与补偿。基准复用§3同格R。定义堆叠差B_alpha=R_control−R0，D_alpha=R_comp−R0，均除sqrt5280；线性预测分别alpha*b与alpha*(A v+b)。不依据第一档结果重新求v，四种完整条件全部按固定计划尝试（身份违规停止除外）。

每个条件均需参数实际可行、全部24输出有限、主细逐包RMS<=1e-8。对照与补偿的细格归一化向量各须满足 `||actual-linear||₂<=1e-7+.1*max(||actual||₂,||linear||₂)`；不符记非线性核验未决，不扩大trust集合或缩小alpha。

每档令 `Z_{q,g}=stack_j R_{q,g,j}/sqrt(5280)`，q为该档的对照、补偿、参考三组，g为主／细两网格。明确 `eta_mesh=max_q ||Z_{q,fine}-Z_{q,main}||₂`；`eta_round=64*eps64*max(1,max_{q,g,j,i}|R_{q,g,j,i}|)`，最大值包含这三组、两网格、24包的全部原220残差分量（按原节点尺度、未再除sqrt(5280)）。`eta=eta_mesh+eta_round`。eta是操作性一致性指标，不是精确ODE误差界。见证通过须两档分别同时满足：

- `||B_alpha,fine||>=1e-5` 且 `||B_alpha,fine||>=100*eta`：搭配扰动确有可分辨的观测作用；否则是低信号未决，不能把本来几乎无变化叫补偿成功。
- 主格和细格各自 `||D_alpha||<=min(1e-6,.1*||B_alpha||)`，且细格在该界内至少留出`10*eta`余量。
- 每包补偿观测差RMS<=1e-5，主细均须通过，防止整体均值掩盖单包变化。

这些数值阈值在读输入前冻结：它们定义本次“近似不变”的操作范围，不是噪声统计置信界。两档均通过才报告“两档近似补偿见证”；一档通过只报告该档有限点证据，主问题仍未闭合为两档见证。全部逐包差和两档结果均保留，不改门选优。

## 5. 下界、解释和停止

本最小方案只提供可行上界与被选点的完整非线性核验，不建立线性最优值证书，也不建立全补偿盒的非线性余项或浮点误差上界。因此没有找到见证、返回目标较大、求解到限、可行性失败、差分不一致、低信号或非线性门失败均不能证明不可补偿。按原因标unresolved后结束；不搜第二方向、不加符号、alpha、起点、半径或更高精度。

见证通过说明quarter附近的该合法beta变化可被本集合内共享c与逐包a/r变化近似抵消；它不把固定x对照称纯beta效应，不证明完整父→quarter有限位移等价，也不证明整个方向连续区间、全部支撑或全局不可辨识。未通过不否定模型族。两档积分只核验所选v的两点，不为整个补偿集合提供任何下界。

## 6. 精确有限工作与资格

每包每网格最多 `1+66*2步长*2扰动=265` 次历史映射；24包×两网格=12720次，RHS为 `24*265*(480+960)=9158400`。完整核验为2档×2条件×24包×2网格=192次，RHS `96*(480+960)=138240`。合计科学历史映射12912次，**RHS9296640次**。这是上限，早停余额关闭，不转移。参考、FD中心复用不再重复求值；任何失败调用实际消耗照计，不重跑。

| 科学工作 | 上限 |
|---|---:|
| 历史映射／RHS | 12912／9296640 |
| 有界线性问题 | 1个，5280×525，外层100迭代 |
| 内部LSMR／总迭代 | 101次／101000迭代 |
| 非线性优化、重初始化、六门、未来轨迹、独立H、SVD、NNLS | 0 |

技术资格最多初查一次、针对明确技术错误修正后复查两次，共3轮。只用纯代数／mock，每轮最多1000回调；真实模型/RHS/ODE/H为0，不生成人工模型轨迹或运行旧科学fixture。每轮至多1个手写2×2有界线性问题，沿同solver设置但max_iter=10、lsmr_maxiter=20，最多11个内部LSMR；三轮技术总上限3个小问题、33个LSMR、660内部迭代，单列不混入科学计数。

固定资格覆盖：x=0、c=beta、beta=0和u上下边界的有限可行性；固定x改变beta时c联动；共享45列与局部块不串包；手写线性／二次映射的中心与单侧两h；主细不一致／非有限／低信号拒绝；实际回存舍入／越界拒绝；同v两档与参考向量比较而非标量损失；求解失败／到限只给上界或未决；映射、RHS和内部LSMR预算守卫；禁止future和初始化选点接口。人工问题验证返回可行和已知可核答案，不用其通过替代真实线性化可信度。

第三轮资格仍失败则不释放。科学执行后若发现源缺陷，先停受影响阶段、保存实际账本，由root审定；无自动科学重跑额度。全部科学与技术成本、wall/CPU、线程、异常和实际调用分开记录，旧余额不恢复。

最终保存身份、差分模板／矩阵hash、全部一致性门、solver轨迹／可行v、两档全部包结果及成本，独立审查后root作代码／模型／科学三决定。默认retain_evidence，default none不变，无L1/L2/L3或新FOM，val/test关闭、TASK4暂停。设计阶段未读原始／私有数组、未执行任何科学计算；本协议也不授权未完成审查前执行。
