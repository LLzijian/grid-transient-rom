# 两种边记忆方程：有限协议

2026-10-21。root采纳实际gpt-6-astra /root/astra6_after_frequency 原案addendum2及独立/root/fom_history_interface挑战的全部实质条件。原案/先前修正/范围反例全部保留；此前BLOCKED建议不采用。旧作业、历史门槛及类别限制不改变。唯一问题：同样100ms观测和512次联合适配下，正弦差记忆与余弦和记忆能否被可观测预测区分？本轮是两个预给定物理假设的比较，不是185项未知公式恢复，不开放真实广泛搜索/项目val/test/P4/TASK4。

## 精确两项和合同

正弦AST={"channel":"edge","expression":{"op":"mul","args":[{"op":"add","args":[{"op":"var","name":"qi"},{"op":"mul","args":[{"op":"const","value":-1},{"op":"var","name":"qj"}]}]},{"op":"var","name":"s"}]}}。
余弦AST={"channel":"edge","expression":{"op":"mul","args":[{"op":"add","args":[{"op":"var","name":"qi"},{"op":"var","name":"qj"}]},{"op":"var","name":"c"}]}}。
规范化只按现有canonicalizer，不用真值身份排序。每项一个共享自由有符号a，初值精确0，无投影。两项均一表达式/一系数；树节点（含var/const）7与5，运算节点3与2。按原规范化AST实核并保存，不删去已登记竞争者。

D=base_D，eta=delta D^T，theta=phi+eta，s=sin(theta)−sin(phi)，c=cos(theta)−cos(phi)，kbar=base_coupling/mean(base_coupling).clamp_min(1e-15)固定，不是待拟合骨架coupling。
额外Hsin=a sum[kbar*(q D^T)*s]；Hcos=a sum[kbar*(q abs(D)^T)*c]。
额外gd分别为 a[kbar*(q D^T)*cos(theta)]D 和 −a[kbar*(q abs(D)^T)*sin(theta)]D；gq分别a(kbar*s)D和a(kbar*c)abs(D)，gp=gpi=0。内部x=(delta,pi,p,q), pi=Momega，g=(gd,0,0,gq)，完整场新增为L^{-1}(J−R)g，L=diag(I,M,I,I)，原端口/完整骨架均参与，不能只比较储能。s/c无量纲，a单位为H-unit/q-unit；q保留原潜在坐标，端口耗散/正定性/零Hessian不作为额外假设。边反向同步D与phi变号，公共角只平移delta。

复用原edge_run01三个residual完整512正弦端点及原冻结train/合成val指标，不重训/重评；从原共同未拟合toy_source和runtime_model种子5101/2/3构造新余弦，禁止热启动已拟合参数。需实际核旧checkpoint_00000共同参数/buffers/flags、GRU、固定source_mass/history_scales、六头0、系数0、CPU/CUDA/采样RNG、Adam组参数顺序和配置；核旧00512计数/trace/模型训练哈希及原fixture身份。无法证明实质相同即关闭，不补跑对照。

仍40状态、30观测、11点100ms一次初始化、自主ODE、无真实q/RHS/干净导数监督。16train弱强各8、未来0.5s；16合成val未来1s，噪声0。新增train-only生成器逐字复现旧SeedSequence([seed,20261019]).spawn(4)的stream0、16/60步coarse/fine；核原clean_sha256[0]/scale exact。禁止cost或fit调旧同时生成train/val的fixture()。只有全部新端点与train选择冻结后才用stream1/110步生成val、核clean_sha256[1]。教师q只用于生成，不外传学习。cost评价用独立非科学16×0.5/1s计时夹具，不碰正式合成val。

训练原完整0.5s归一化MSE，固定train尺度及floor1e-6，batch12=rng.permutation(16)[:12]，原采样SeedSequence([seed,7])，float64 CUDA RK4.01。Adam原encoder3e-4/closure1e-3/skeleton1e-2、残差六头.01，原其余Adam配置与顺序逐字核，clip100。所有允许骨架/闭合/GRU/六头联合更新，恰512；stage0/每32共17检查点；原多项式/phi/尺度/单位不改。无恢复/重试/新端点/延长。

## 前瞻预算与准入

仅先从P0余8122.771089登记2200：static150、唯一QA600、唯一cost600、support hold700、metadata150，分项不挪。planning600已在P2另计，不重复。登记后P0未分配5922.771089；父累计charged/held先134972.46971589996（原134272.46971589996+support700），未执行最大1500另留，父未分配122727.53028410004。P1余3982.090377尚不扣3300、不启科学。其他类别不互借，旧hold保留。

QA/cost各native600/PS570/self540，完整费用含所有子进程一次。static独立实现审查及必要静态纠错在QA前；QA仅一次，无QA02/执行后修复。新专用experiment/test与BASE私有小监督助手，不改任何历史绑定代码，不造通用平台。所有实际命令/源绑定/环境/输入身份先归档，核真实PID/整数FILETIME/GONE/锁，禁止重复队列。

QA真generic参考须绕过快速映射，规范AST解析额外储能后对内部状态求导，独立组合完整J/R/B场；检查CUDAfloat64完整场、原0.5s目标、全部原始梯度、clip/Adam、非空Adam/CPU-CUDA-NumPy状态恢复下一更新、非单位M/边方向/公共角/对象唯一/缓存/导出/一次初始化/未来拒绝。冻结容差field rtol2e-11 atol1e-11，loss2e-10，grad/Adam2e-9；不得放宽。真实旧合同零更新核验属于QA准入前段，未过不执行后续更新或cost。正确性夹具不读正式val。

唯一cost在QA/合同通过后：每seed恰4次完整train更新计算，1→参考2→恢复1重做2→3，共12，状态全部丢弃。step1非空snapshot为cost-only，不改旧restore对32倍数限制；正式science仍0/32/…512无恢复。测完整同步update最慢值、同格式保存/恢复、load/export及峰值显存。train估价每seed=1.25*(512*t_update_max+17*t_save_max+t_load_max+t_export_extra+t_supervision20)，save含export则extra0；原合同额外3次全train里程碑评分128/256/512计入t_extra（另加，不漏掉）。全部互斥明列。非科学16train+16val等长双步长完整评价/写盘，加同样20监督秒后×1.25；每seedtrain<=640、eval<=40；三组及下面完整P1帽可容纳才准入。科学成绩不从cost提取，不用旧2634直接替代新上界，不按最快seed/平均计。

失败/数值/超时/合同差异/费用不过即关闭支路，无重测/换项/缩种子/缩512/少事件/取消数值核验，不以实现失败否定物理家族。原失败证据保留，工程不默认采用。

## 条件性P1完整阶段

P0全部通过后root另写接受与P1登记：总3300=3train各native700/PS670/self640共2100+3eval各native100/PS70/self40共300+metadata100+support800。按实际P0关闭账本取消未花分配后再扣，不转类别。顺序5101→5102→5103，各新512从未拟合来源。科学执行前新源/协议/实际命令/旧合同/admission全归档。

所有新512有效端点统一冻结后，计算完整16train MSE和逐种子选择表，与旧正弦train比较；同精确MSE按树节点较少、独立系数较少、规范AST字典序破平局，不用val/真值选择。选择表及源哈希归档后才生成/评价新cos正式合成val。每seed各一次完整train+val双步长，保存所有16逐事件、弱强分项、512曲线/系数/源，不删失败事件；有关总体未决保留null。缺端点不称三seed完整。

原合成val MSE<=.01、双步长归一化max<=1e-3仅定义该候选在此条件下的预测可行性，原完整发现门槛不改。两个都过可保留两种有限可观测解释；一个不过只说明原初始化/目标/512下的差异，不证明唯一真机制或坐标等价。旧val已知、开发复用，不称盲确认/真实电网泛化。三种子不是增加独立事件数，不加显著性阈值。原真实test/val关闭、TASK4暂停，P3次数/P4门槛不动。

结项经未参与实现的独立结果挑战，root决定范围与科学解释。合格必要工程可有限PR合main并核expectedhead/远端；若完全复用代码仅归档，不为合并造修改。无真实电网模型晋升/default。各结局归档关键源/正负结果/命令/费用，保留大资产路径哈希。父研究未完成，结项后按真实余额核下一合法分支并实际Astra规划，不以next_action代替作业。
