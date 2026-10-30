# Campaign：力记忆残差的可解释归属

状态：**active**（2026-10-30 起）；性质：Campaign 卡，不是实验预登记，不授予执行权限。
范围：train fit297 与 100 ms 部署合同下，当前 40 维力记忆族的残差主要来自哪一类机制。
架构版本：`memory_transport` A/S/P（来源 N_M192→SGM A_28311 step96，A96）；下一候选
E=κD（见 [Astra 条件准备](../reports/astra6_next_structure_preparation_20261029.md)，DRAFT，未采用）。

## 假说与竞争解释

主假说 H1：当前族的共同模态闭合不足——`1ᵀQ=λ1ᵀ` 使 `d(1ᵀr)/dt=κ1ᵀω−λ1ᵀr`，
边权与零行和转运都不能改变这条标量方程；只有 κ、λ 或改变标量闭合的储能结构（E）能改变共同模态响应。
竞争 H2：100 ms 历史对初始隐力 r₀ 的可辨识性不足，误差由未观测初态主导。
竞争 H3：128 步 Adam 的优化机会未用尽，模型间差异小于优化/种子散布。

三者不互斥；本 Campaign 首先要求排除至少一种，并把仍未决的部分写清楚，而不是再选一个更大模型。

## Run 树（本 Campaign 收录的已闭合 Run）

| Run | 目的 | 结论 | 材料 |
|---|---|---|---|
| force_graph_20261028 | 空间作用进入静态耦合还是记忆传播 | 传播 M 相对静态 S 有稳定风险/8 s 漏判收益，保留为研究起点 | reports/force_graph_20261028_conclusion.md |
| force_exchange_20261028 | 单参数 p–r 互易交换 | 未满足父模型稳定精度保护，未保留 | reports/force_exchange_20261028_conclusion.md |
| state_gated_memory_20261028 | 角差调制 | NOT_RETAINED；A96 保留为失稳风险候选 | reports/state_gated_memory_20261028_conclusion.md |
| weak_force_identification_20261029 | 消去隐力后一次直接辨识 | λ=0，CANDIDATE_NOT_ADMITTED | reports/weak_force_identification_20261029_conclusion.md |
| memory_transport_20261029 | 固定平面保守转运 S 与耗散路径 P | 共同 128 NOT_RETAINED；无延长/第二种子 | reports/memory_transport_20261029_conclusion.md |
| closure_probe_20261030 | 残差四分解 + r₀ 条件数 + COI 预算 + 64/128 | H2 不支持为主要解释；H1 降级未否证；H3 强支持；H4 开放 | reports/closure_probe_20261030_conclusion.md |
| checkpoint_selection_20261030 | 九点完整关键事件与固定时机选择审计；optimizer 0 | 无全门点；step16稳定收益与实际区间代价、step96实际检出收益与8 s新增漏判均保留；不晋升 | reports/checkpoint_selection_20261030_conclusion.md |
| p_port_diagnostic_20261031 | 冻结p坐标的来源、滤波、共享常数与历史反推诊断 | M2混合来源；平衡基准差0；反推块未过冻结数值门，M3归因unknown；0optimizer/0新预测 | reports/p_port_diagnostic_20261031_conclusion.md |
| p_reverse_precision_20261031 | 已失败反推门的数值来源/活跃调用/初值敏感性 | 当前Torch64逐位复现旧buffer差；297事件递推/QR/场差核对通过；不追认父M3，无采用 | reports/p_reverse_precision_20261031_conclusion.md |
| analytic_p_backward_qualification_20261031 | 两套固定权重的解析反推完整数值行为资格 | 两候选各297事件/2790区间全通过；分类身份不变，原科学准入均false；仅显式数值入口，不晋升 | reports/analytic_p_backward_qualification_20261031_conclusion.md |

## 当前认识与未决

已知：A128 在稳定 5/8 s 风险上弱于 R_N（0.867537/0.625571 对 0.830421/0.591762），
在失稳真实区间风险上强于 R_N（3.045603 对 3.512358）；S/P 相对 A 无登记收益；
A128 的 64→128 面板 5 s 风险回升超过原 105% 保护线，说明优化路径敏感，但不能单独证明欠优化。
closure_probe_20261030（probe，0 optimizer）已闭合：r₀ 条件数 1.613、ridge 贡献 1e−5、ridge 敏感 3e−5，
H2 不作主要解释；COI 预算中 Δr 占 75–77%，但标量闭合残差主要被 W1 轨迹差异解释（放大倍数 1.02–1.09），
H1 降级、未否证；全队列 128 比 64 差，九点曲线非单调、最小值在 step16（5 s 0.752083、8 s 0.576173，
两项都优于 R_N），**H3 强支持**；p 通道约 1/3 误差来自固定 p 动力学（探索性），H4 开放。
未决：共同模态闭式的确切必要性（需不依赖机械反演的检验）；p 端口最小扩展；失稳窗口过短。
已排除：在本轮范围内，固定平面转运 S 的 40 维端点没有带来登记收益；这不否定一般非互易记忆族。
checkpoint_selection_20261030已完成：9点无全门点。step16稳定5/8 s风险0.752083/0.576173，
实际区间风险4.096469、漏判22/30；step96实际区间风险3.152346、漏判20/30，保原9检出并修复1个，
但8 s相对A96新增1个漏判，仍4/30。全部数值资格通过，来源健康/兼容已对账，优化充分性unknown。
固定T为9，缺覆盖点不缩分母。单一已消费train路径支持权衡观察，不是独立确认或A族否证。
最新p端口证据：M2为转速误差传导与固定p残差混合；原反推精度失败已定位当前可复现算法差，
解析数值候选在完整登记范围内保持原行为。旧M3 unknown和上游tm含义限制保留。
下一有判别力的问题：在已合格显式数值入口上，用新的前瞻设计区分共享p常数不匹配与方程形式误差；
不以旧已暴露β作新确认，不因数值资格启动训练或改状态。E/s暂不作为首选。

## 保护与边界

R_N 保护档案、A96 来源、各 Run 端点与失败证据、已关闭余额均不因本 Campaign 改变；
原 val/test 关闭、旧已消费 test 永久退出、新 FOM=0、TASK4 暂停、default none。

## 2026-10-31 自动推进更新

用户已要求按harness开展实验并自动迭代，既定方向有界科学执行恢复，每小时本聊天接续。
已完成p端口诊断：来源297/297、base.pm_star与来源平衡Pm差0；M1自身闭合全通过，
M2稳定和失稳队列均为ω传导与固定p残差混合。反推buffer重建最大差1.5673e−10未过1e−12门，
M3十次离线共享参数估计均未形成归因资格，结构建议为空。未修改旧门、模型或数据。
上游tm为转矩字段的补证限制物理功率解释；当前结果只按冻结p坐标使用。
下一独立probe定位反推误差来源/真实调用路径/影响；设计可先准备，执行必须等待本轮归档完成。

反推精度probe已完成：固定binary64输入下当前Torch64逐位复现旧buffer，相对舍入高精度参考
差1.5673e−10，SciPy64对同参考差3.469e−18（对未舍入MP120差1.06285e−16）。
两模型两历史步长均实际使用buffer；297事件上p历史/QR/r₀/初始导数的传播核对全通过，
maxΔr₀约1.306e−9、maxΔωdot约8.612e−9，长期预测影响未评价。
原p诊断失败门/M3unknown保留。下一解析反推数值候选资格草案仅DRAFT_UNADOPTED，
最多两个固定候选，数值资格与原绝对科学门/固定T_anchor分开，不晋升原不合格检查点。

## 2026-10-31 解析数值入口资格结果

analytic_p_backward_qualification_20261031已执行正常两份paired，0optimizer/0拟合/0新FOM。
每个候选297/297事件、2790/2790必要区间合格；最大COI归一化RMSE为5.19124e−9/1.91852e−9，
五项主指标新旧差均在各自原主细误差和内。两时域完整分类ID集合一致，fine/coarse分类与窗口变化0。
SGM A96解析候选8 s/真实区间漏判3/30、21/30，MemoryTransport A_30311 step16解析候选4/30、22/30，
均与各自父版相同，两时域稳定误报均0/267、数值未决0。T_base9/8完整覆盖，科学T_anchor仍9。
两者implementation_qualified=true而science_entry_eligible=false，数值实现资格没有改写原科学门。

代码retain_branch，允许后续冻结协议显式引用两个合格overlay身份；模型retain_evidence，四指针/default none不变；
科学continue，下一投入回到p形式归因。旧反推失败与M3unknown不回填；单一已消费train上的数值资格不扩成物理确认。
19项人工测试，唯一prepare及各候选一次paired、summary一次退出0；真实修复0、额外预测未释放且余额关闭。
原生wall：prepare25.079 s；两个候选228.938/192.738 s（并行重叠228.944 s，单项和421.676 s）；
summary4.464 s，CPU总进程秒unknown。最终审查与远端归档核验后进入下一Run；不重复已完成Job。
