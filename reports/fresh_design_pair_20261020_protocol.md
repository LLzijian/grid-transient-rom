# 剩余384条件的新固定结构比较：root登记

2026-10-20。root采纳本轮实际gpt-6-astra代理/root/astra6_after_fom_environment_failure的完整[原始方案](astra6_after_first_physics_QA_20261020.md)。此前首次物理档严格关闭，归档ef0aecfb18a28d79345510fcb6b040c05ea466c5；其原设计复用例外永久关闭。本轮使用新的384条件池，不调用旧待执行命令、不改旧源或证据。原test关闭、TASK4暂停、结构生成搜索关闭、默认模型不变。

研究问题与原active_fom_pair科学协议保持：固定系统/40状态/100ms单次初始化下，full8/delete7预测取舍能否由新激励辨别并在共同适配后复现。原train、全部历史新train尝试、旧val条件及旧设计全部48候选+24val按唯一物理键排除。新池逐格计数严格采用本次Astra表，总384、三层104/176/104；不一致停止。一次PCG64种子7301抽24val（每母线1/每层8），7302抽48候选（每母线两不同层/每层16），7303预锁随机12（每层4）。复用整数DP纯函数，保存精确完成数/边际/抽样路径；Zval4252621230，候选Z1097450640或399072960，不用原prepare、不重抽seed。新72与所有排除零交集；统计目标为这384剩余池上的约束设计，不宣称每母线每层1/3或主动优于随机的因果效应。

固定BASE为private_data/closure_mechanism_20261019/P3/fresh_design_pair01，campaign=BASE/campaign01。新增experiments/fresh_design_pair_20261020.py与配套测试，直接复用冻结旧acquire/fit/DP，不套失败wrapper。私有launcher在parse后把全部路径resolve结果赋回参数，验证与转发使用同一绝对值；PS绝对路径规则不放宽。required binding显式包含config，启动前断言。所有实际worker命令路径为绝对路径；QA故意从相对launcher CLI输入开始，以同一生产launcher→nativePS→真实QA worker证明规范化、PID/FILETIME/argv/cwd、内外exit0和清理GONE，再实际执行有限接线与设计检查，不把AST/定义测试记为成功。不另开环境doctor、P0或科学计时探针。

fit.CAMPAIGN仅在当前进程限固定新BASE并恢复，源模型限原六个P2 global1024完整模型/Adam/RNG，新端点限BASE/models/<key>/checkpoints/step001280；新数据/预测/schedule/输出/val freeze resolve后限本档，freeze文件放campaign内。科学入口与每条件边界使用原真实期望环境preflight；共同环境/source/case错误真实穿旧Exception捕获停止批次并保留剩余not_attempted，真实条件物理/数值失败保留、不重试补位。旧scientific manifest兼容格式保留，新入口/协议/命令/环境在独立外层绑定。

前瞻锁定40000秒P3：父累计61197.1149751、余额198002.8850249，登记后父未分配158002.8850249；P3余85108.0596498，登记后未分配45108.0596498。其余类别及旧900hold不动。类别限：QA/准备1000、48短历史1500、D/15秒事前预测750、完整train3000、六模型256含schedule18000（每模型所有attempt3000）、train评价/谱/导出3125、完整val3000、val评价/冻结/统计1875、支持3250、非科学退出归档2000、不可执行重复预占2500。唯一外部中断恢复≤31步、≤2500嵌套于各模型3000/训练18000；NaN/确定性失败不恢复。失败/启动/清理累计，子包络不双计，类别不互借。

科学顺序与全部门槛继承原协议：真实QA/绑定准备→48双步长100ms（真实11点/绝对时刻/系统一致、scaled RMS≤.001、分类一致，unknown不改stable）→六源5秒500点评分、ROM .005/.01均D≥1/B≥2.5e-5→随机12+主动12、每层不足4即关闭→完整FOM前24×六源15秒预测冻结→24双步长完整train（资格按FOM/系统/因果历史及真实区间，不按ROM误差删样本）→原366+共同新train六模型各追加256→各train评价/方程/39维局部谱→六个global1280及统计冻结→24新val双步长→一次全六评价。三seed6101/2/3，新采样7101/2/3，1/2/5秒86/85/85，LR .003/.0003/.00009、原损失/force/float64保持，每32保存，256唯一端点。

val仍按设计条件性稳定5秒加权风险（.8均值+.2最高10%权重尾均值、边界分数权重）及原ceil风险分列；主seed6101，delete7−full8，逐事件和三seed方向、频率/功率/失稳损害独立报告，无新增显著性或bootstrap。确认任何未覆盖/数值未决则全24未决；六端点不齐不生成val。无模型晋升或L2证书替代。

QA与准备累计1000、最多一次明确修正重验；任何失败均保留真实回执（无native时单列实际工具包络，不伪造）。若此新档QA再失败，关闭并暂搁这条启动实现路线，root作实质路线评审，不再排掉72条件开同构档。源/环境/实现确定性科学失败按阶段关闭，不借恢复/新编号重试。有限独立实现审查与精确命令/源归档后才执行；QA通过立即一次设计冻结并启动合法短历史，不加代理诊断链。各阶段实际结果均选择性GitHub归档、结项独立挑战；合格工程才PR合并，模型仍须原科学门槛。本登记无当前科学PID。

新入口及私有launcher已封版，代码来源743fcbee73e59a7753300e07502b03d28eba0590；[独立同轮实现审查](fresh_design_pair_20261020_independent_review.md)支持源/命令冻结后实际QA，root采纳并保留其freeze文件必须在campaign内的条件。QA为GPU解释器`-m experiments.fresh_design_pair_20261020 qa`，worker --output/--wrapper-output均为BASE/stages/qa01绝对路径，自限240、原生外层300。root对launcher的command/output/worker/binding/prior故意使用相对CLI，必须在真实回执中看到全部规范为绝对路径、实际worker存在且完成17项检查。准备为prepare-binding、BASE/stages/prepare01，自限120/外层180；仅QA通过后使用实际QA回执建立prior并冻结调用。精确命令/hash及显式绑定见结果preparation；代码来源与执行前命令归档提交分别记录，不声称二者同一提交。本条记录为待执行命令，无虚构PID。
