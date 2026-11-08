# analytic_p_backward_qualification_20261031：Scientist 最终有界设计

**本轮检验解析反向p传播能否成为保持原行为的数值实现。** 固定A96/A16两套既有权重，仅更改p_backward；完整因果、数值、逐事件与时机资格通过之前不采用。小初值差不是长期预测资格，数值实现采用也不解锁旧M3。本设计供root冻结协议和独立审查，本任务未执行实验或读取数据载荷。

日期2026-10-31；task_id=`/root/scientist_checkpoint`；保留spawn配置 `requested_model=gpt-6-astra`，`server_resolved_model=null`。新Run建议类型probe（数值实现资格），父Run `p_reverse_precision_20261031`；root已核父Run远端归档be39cfb，实施登记仍须绑定其真实archival_verified、源码/协议SHA及审查。沿既有自动迭代授权，不由旧暂停快照阻止；本轮工作量与授权边界如下。

## 1. 固定候选、资料与不变项

唯一候选A96_analytic、A16_analytic，分别以SGM A_28311 step96、MemoryTransport A_30311 step16为父；引用父probe及checkpoint_selection的同一model哈希、事件manifest和完整来源。原模型、protected anchors及历史导出保持只读，新数值候选另有身份，不覆盖原版、不改变working_candidate/incumbent/default none。

唯一允许改变的state_dict项p_backward；g/a/S、可训练权重、ODE、40维状态、100 ms因果输入、未来积分器与步长均不变。数据只用train fit297（267稳定、30失稳）；val/test关闭、新FOM=0、TASK4暂停。optimizer=0、共享参数拟合=0；不增加S/P/R_N数值候选，不扫阈值/步长/参数。旧β已暴露，若引用仅作冻结既有估计，本轮不重拟合、不用于改模型。

父probe已确认当前Torch64逐位重现旧buffer，误差可沿反推/QR/初始场传递，历史唯一生成原因unknown。旧反推1e−12门失败和旧M3解释unknown保留；上游tm物理含义限制不消失，本轮p仍按冻结数值坐标解释。

## 2. 独立数学参照与原入门门

唯一构造：用加载后float64常数按原顺序装配Q=−.01P，以Q的精确binary64值作数学输入，逐机3×3解析式80位计算、120位独立核验后单次舍入binary64得到B_ref。每机块[[d,b,0],[0,0,c],[0,0,0]]首行 `[exp(d),bφ₁(d),bcφ₂(d)]`；φ₁=(exp(d)−1)/d，φ₂=(exp(d)−1−d)/d²；d=0精确取1和1/2，禁止衰减下界或算法择优。

80/120位解析与独立高精度expm差≤1e−60×max(1,maxabs参照)；舍入后B_ref对独立float64参照须通过原atol=rtol=1e−12。高精度只用于一次buffer准备，实际初始化与未来仍float64。相同常数可复用已核数学回执，但须重新核输入Q、输出B_ref和实现SHA完全对应，不能仅复用“PASS”字样。

沿前Run固定零a/常值/线性ω/正反向人工套件；对实际297历史，候选history_system的p_history与独立解析反推须通过原1e−12门，历史主.0025/细.00125分别核对。旧buffer失败不回写为成功。任何数学参照或历史入口不通过，停止该候选后续预测。

## 3. 新候选格式与严格加载，不修改冻结科学源码

采用专用诊断格式 **analytic_p_backward_overlay_v1**，不冒用旧model格式。其最小字段：候选ID、父模型路径/SHA/原format、父源码闭包SHA、原p_backward SHA、Q与B_ref SHA、B_ref的float64形状/值、唯一patch键p_backward、解析实现版本/参照回执SHA、协议SHA。候选身份由父模型SHA和overlay SHA共同组成；父权重不在overlay内另造可独立改变的副本。

专用loader顺序固定：①验证allowlist中的A96/A16父SHA及原格式；②调用**未经改动的原严格loader**加载原版并通过其所有config/buffer/plane校验；③验证overlay及数学来源与加载g/a一致；④deepcopy原对象，仅copy_ B_ref到p_backward并清缓存；⑤核除p_backward外全部state_dict逐位相等、训练/梯度设置不变。原loader任何断言失败即停止，不能跳过、放宽或改config迎合候选。

A16原MemoryTransport.from_export会将fixed buffers与由config构建的模板逐位比较，因此不将改B后的对象直接export为旧格式再要求原loader接受。候选保存/重载始终走专用overlay loader，旧父模型仍走原loader且保留旧B。候选的加载—保存overlay—重载须保持B_ref/全部权重逐位一致；另核原对象前后哈希不变。候选缓存与评价必须绑定复合身份，禁止自动退回父loader后丢失patch。

验证实际initialize→encode→history_system调用新buffer、末点δ/ω/p硬钉相等；11点δ/ω和末点p₀之外的信息不进入初始化，改变非末点历史p占位输出不变。合法空/热缓存初始化按原容差一致。只用诊断包装/新格式实现，原冻结模型、sgm科学评分模块及旧loader源码不改；若无法实现则technical_invalid，另提设计，不临时更改架构。

## 4. 两候选完整新paired资格

原A96基线复用其原fit000096完整缓存；原A16只复用checkpoint_selection_20261030/step000016完整回执绑定缓存。冻结前核model/prepare/精确事件ID/预测/evidence/complete全部哈希及数值资格，并提取身份集合与T_base；基线缺失不自动重跑或换对象。

正常工作量为2个候选各1次完整fit297 paired评价：CPU float64、与原评价同批次口径，future .01/.005 s、history .0025/.00125 s各自初始化；用原sgm.evaluate指标/分类函数，接收专用loader已构建的候选对象，确认deepcopy不丢patch。新自主预测按原15 s缓存范围生成，稳定1/2/5/8 s评分；失稳 `n_e=min(1500,len(x_e)−11)`，fine[::2][1:n_e+1]对x[11:11+n_e]，不补未来、不补零、不删事件。

全部原资格保留：必要细子步/粗步有限，COI与共同转速归一化主细RMSE≤.001；分类一致，双方检出时确认时差≤.02 s；所有必要评分行合格。任何数值未决保留30/267及全297分母，阻止候选通过。报告原risk、共同ω RMSE(rad/s)、归一化p RMSE及完整分类，不把risk叫RMSE或把p称已独立认证SI功率。

## 5. 新旧配对门与精确定义

每个候选仅与自己的原版作实现行为保持比较，分别比较fine对fine和coarse对coarse，均对齐.01 s。稳定1/2/5/8 s与失稳完整真实区间逐事件核COI/共同转速归一化差≤.001，另核p归一化RMS差≤.001；8 s分类完整路径还须沿原分类确认端点的跨实现数值检查，不能只查失稳短记录。全部必要输出有限，不用截断缩小差异。

对每个原主指标j（稳定5/8 s risk、8 s共同ω RMSE、8 s归一化p RMSE、失稳真实区间risk），定义 `u_score,m,j=|score_fine,m,j−score_coarse,m,j|`，m分别为原版/候选。主值仍fine；门为 `|score_fine,new,j−score_fine,old,j|≤u_score,new,j+u_score,old,j+1e−12`。粗值用同一冻结真值、队列、时间范围和指标函数从缓存计算；RMSE不得使用risk扰动替代，任一值null即未决。该门是原数值分辨范围内一致性，不是统计等效证明。

8 s与真实区间两时域分别冻结原版的合格检出、漏判、误报、数值未决、未覆盖及判据未判定ID集合，候选须逐集合一致。完整原分类category及window_settled变化也列出；若影响上述集合则不通过。不能只比数量，不能用新增检出抵消新增漏判。门外的行为变化可记录为观察，但本轮不改门将其称作改善。

固定 `T_base,m` 为每个原版真实区间内数值合格且truth/fine/coarse时刻均非null的全部ID；新候选须完整覆盖，truth时刻逐事件一致，T空不通过。定义 `v_time,m,i=|t_fine,m,i−t_coarse,m,i|`、`u_time,m=max_i v_time,m,i`、`E_m=mean_i|t_fine,m,i−t_truth,i|`，三者单位秒且只在固定T_base计算。

时机门同时要求 `E_new≤E_old+.01+u_time,new+u_time,old`，以及每个i的 `|t_fine,new,i−t_fine,old,i|≤.01+v_time,new,i+v_time,old,i`。任何缺失不得缩小T后重算；时刻沿原确认窗口末点及0.01*(index−10)定义。不用u_score替代u_time。

## 6. 原科学门独立保留

原A96/R_N逐时域检出并集、固定9事件T_anchor、原绝对risk/漏判/误报/共同ω/p保护目标照旧另表计算，绝不从T_base替换或缩小T_anchor。A16原缺一个T_anchor事件，数值实现行为保持通过不等于科学准入、模型晋升或补齐覆盖。A96/A16原能力边界及step96的真实20/30、8 s 4/30权衡继续保留。

数值候选的数学入门、原因果/主细、跨实现指标/身份/T_base门须全部通过才记 implementation_qualified；原科学门单列 science_entry_eligible，不互相替代。本轮模型不晋升；不得因qualification为true自动改变任何默认/受保护指针。

## 7. 有限停止、成本与确认

最大2个固定候选、正常2份新完整paired评价；数学/因果/overlay加载套件一固定套。身份、数学、因果入口失败停止其依赖阶段；科学/数值门失败不触发改步长、换候选或放宽阈值。仅纯实现错误允许全Run至多1次有记录修复，重跑受影响资格；若已生成预测，最多重做受影响的2份，因此正常+修复总上限4份新完整paired评价，科学候选仍仅2个。不挪用父Run关闭余额。

失败产物、修复理由、旧新源码/协议哈希和重复成本全部保存；未知作业先对账，不盲目重放。正常执行、纯修复与科学门失败分别记录；单个失败不阻止另一独立候选完成，但通用数值实现采用须两者都合格，不能只报道通过的一种。

成本单列数学buffer构造、overlay加载、初始化、积分/输出、总wall与修复；旧基线缓存只引用其原推理计时，不记入新增计算。只有同设备/精度/步长/时域/批量/事件范围可核对时才比较同精度成本；否则记未比较。无额外计时搜索，不从矩阵更准或少量耗时差宣布加速，缺失成本null。

确认要求独立数学参照、全部因果/序列化/主细/新旧配对/身份/T_base门及独立只读审查。对固定train的完整行为保持证据可支持数值实现资格，不是新数据泛化、物理正确性或机制确认；父probe的小初值差不能替代本轮任何门。执行前根协议、实现与输入manifest冻结，完成后审查、三个决定与远端归档核验，再考虑下一Run。

## 8. 三个决定与后续边界

代码：两候选全部implementation_qualified且独立审查无未决时，root可采用显式版本化数值构造/overlay实现；旧模型加载仍保留旧B，不做隐式迁移。模型：retain_evidence，各候选单独身份，受保护原版和default none不变；无科学晋升。科学：说明原行为是否保持及失败范围，原优化权衡、旧反推门失败、旧M3unknown与单位限制不回写。

任何后续p形式归因必须另立前瞻协议，绑定已合格实现与来源；旧β/残差明确属于已暴露估计，不能当新独立确认，不能用本轮实现资格倒填旧M3。是否受约束适配p常数、增加状态或继续优化由新科学问题决定，本轮不自动推进到这些改变。
