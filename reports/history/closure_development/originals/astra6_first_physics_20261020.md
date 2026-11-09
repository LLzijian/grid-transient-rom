# Astra6：固定设计的首次物理求解与完整六模型比较

2026-10-20。实际规划模型 **gpt-6-astra**，代理 `/root/astra6_after_fom_environment_failure`，本次为环境工程验收后的新科学轮次规划调用。本文为完整原始返回，root另行记录采用决定。本次只读协议、源码和已有环境QA/费用证据；未导入科研模块、求解、训练、读取轨迹或权重、读取旧val/test，未递归委派。只新增本文；沿用已读取的anti-defensive-writing原则组织报告。

建议root登记并执行一个40000秒P3完整档：在环境接线通过后，沿用原字节冻结设计，完成48条件双步长短历史、随机12＋主动12的完整train、两结构各三seed追加256更新，以及冻结后24条新val的一次比较。问题仍是在固定系统及激励设计下，full8与delete7的预测取舍能否由真实新观测区分，并在共同适配后复现。结构搜索关闭；本轮产物为研究候选及固定设计条件下的机制比较证据。

## 依据、已通过条件与历史界限

科学定义完整继承reports/astra6_active_fom_20261020.md、其design_amendment、reports/active_fom_pair_20261020_protocol.md；本文只明确新阶段身份、已通过的环境、一次条件复用政策及执行接线，未修改原科学门槛。权限来自AGENTS.md、docs/engineering_loop.md与父持续任务授权，预算按现有cost_snapshot，不由本文增发。

原档48条件、96次调用在ANDES导入前失败，0物理历史、0D、0完整train、0更新、0val；归档c3443ed7ddf68ed0b0ccc06e1db027e91c39f54d和3312.2232031秒闭账不变。依据原conclusion、short_result、short_attempts与原始回执；不能以exit0改判成功。原18组GRU、P2六个1024端点、首次D及P0短历史探针均不重跑。

环境工程按reports/fom_environment_repair_20261020_protocol.md完成，结论及验收审查已记录；真实回执为private_data/closure_mechanism_20261019/repair/fom_environment01/native_qa01/execution.json。唯一QA外层17.84752110001864秒，90项source/config前后一致、内外exit0、cleanup完成且worker GONE；真实FOM attestation和spawn一致、GPU CUDA可用、9项保护通过。加600支持闭账617.8475211；无QA重验。**以实际探测Python3.12.14为准**，此前静态pyvenv.cfg的3.12.13记录不代替运行时身份。

许可FOM exe为E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe；GPU exe为E:/03-nn_Transfer_Func/kiki/.venv-gpu/Scripts/python.exe。ANDES2.0.0、NumPy2.5.2、h5py3.16.0，GPU Torch2.13.0+cu130、RTX4060Ti；case SHA256 9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5，train公共物理指纹4da835555c89e36b4bb8f0a8e5f7edfce94b6207a38e028f57f34ded46de7b72。后续每阶段仍实际调用experiments/fom_execution_environment_20261020.py的同一preflight，并与本次QA的完整身份比对；QA通过不取消生产核验。

## root须在执行前采用的一次窄政策

准确名称是 **“相同物理条件重新执行、首次物理求解”**。原no-FOM-retry和失败条件排除条款不能解释为原本允许重试。root须显式前瞻登记唯一例外：仅对已经证实统一import ANDES错误、没有进入求解且没有产生任何新科学成绩的原96项启动失败，在合格环境修复后允许这一次完整科学执行。原档不重开、原失败不改判，软件尝试次数与费用分别累计。科学问题和阈值不变，不把原96次失败说成不存在。

reports/fom_environment_repair_20261020_independent_review.md已对这一政策给出有条件支持；root必须确认实际新实现与该审查条件一致。例外不涵盖任何真实求解后的数值失败、物理失稳、历史不足、D不足或训练失败，也不自动涵盖新一轮环境错误。再次确定性启动/实现失败不得再引用本例外启动第三次。

复制原campaign01的design.json和val_conditions.json，**字节与顺序原样保存**，不重新调用prepare_conditions或重新抽样。原design SHA256为c475a3bf11aaa17e2a3ca4642291a43d863541be14ec0fefdec5cd072de8c126。新provenance单独关联旧48候选/96错误、随机12、旧24val设计及其哈希；不在旧设计内加字段导致原字节改变。24val尚未生成，因此保持原先冻结条件；不称其已有独立确认。其余历史失败和旧val条件继续按原规则排除。

唯一候选物理条件仍48；报告分别列出旧48条件的96启动失败与本轮48条件最多96步长执行，不能合成96个独立条件。没有新分数参与设计选择，所以沿用同一设计不会引入成绩驱动的重抽；结论范围仍是原受排重约束的设计分布。

## 固定工作目录与最小实现

仓库工作目录固定 `E:/03-nn_Transfer_Func/kiki`；本轮BASE固定 `E:/03-nn_Transfer_Func/kiki/private_data/closure_mechanism_20261019/P3/active_fom_pair_first_physics01`，科学campaign为BASE/campaign01，阶段回执在BASE/stages及BASE/native_*。启动前不存在任何本轮科学输出；旧active_fom_pair01保持只读。固定路径由root在协议、命令JSON和wrapper内同时绑定，不接受任意目录扩权。

建议新增一个小wrapper `experiments/first_physics_pair_20261020.py` 和本轮私有launch_stage.py，复用原采集/fit源码。旧active_fom_pair_20261020.py、active_fom_pair_fit_20261020.py、旧launcher及其source_manifest不编辑。新wrapper有prepare-binding、acquire、fit三条明确分派，不创建通用运行平台。

1. **prepare-binding**只复制原两个设计文件、核验hash与96条失败谱系，建立本轮source/环境/费用/命令身份；不重抽、不读科学轨迹或反序列化checkpoint。原source_snapshot/verify_sources保留原格式和验证语义。在新campaign保存重新取得但格式兼容的科学source_manifest，design_artifacts指向新副本；原source、旧协议、原checkpoint精确身份仍可作为谱系绑定。新wrapper、新协议、新launcher、环境完整身份和实际命令以外层独立binding绑定，不能随意往旧manifest加键造成verify_sources必然失败，也不能通过关掉verify_sources解决。
2. **acquire**在任何条件前按类别调用已验收guarded_call/preflight：short/train/val使用FOM exe；score使用GPU exe。随后传明确--campaign、--output、--source-manifest、--source-commit、--max-seconds给原采集main。wrapper传递参数可用局部sys.argv替换并恢复；模块导入本身不得触发任何科学步骤。新private launcher必须接受新wrapper的命令结构并严格验证，不能仍限定旧入口或全GPU exe。
3. **fit**使用GPU exe。静态确认旧fit.CAMPAIGN当前硬编码旧BASE，load_events用它约束新数据与val冻结路径，直接调用会拒绝新目录。wrapper在当前进程中明确将fit.CAMPAIGN设为固定新BASE，然后调用原fit.main；这只是将原同样严格的目录边界迁到新档，不能移除is_relative_to检查或改成整个private_data。调用前核验原常量仍为预期旧BASE，保存before/after绑定证据；退出时恢复该模块变量。训练spawn若会重新导入此模块，必须由同一wrapper入口重复固定绑定，并用合成spawn检查，不能假定父进程变量自动传给子进程。
4. 在fit调用前，wrapper额外严格检查events-spec、new_directories、freeze_path、output、schedule、stage-identity、模型路径及预测清单：本轮生成物必须在固定新BASE内并匹配train/val目的；P2 source-output/source-identity仅允许原六个注册1024端点；旧366 train只能经原许可train_cohort加载；旧active_fom_pair01仅可读原冻结设计与失败身份，不得成为新训练/val数据入口。val模型必须是本轮六个global1280端点。拒绝旧档freeze、新旧val混用、路径穿越和symlink逃逸；所有路径先resolve。
5. 每阶段先写真实worker身份和环境preflight证据，再启动科学callback。原fit某些模式较晚才写worker_identity，wrapper应提供单独可被原生监督观察的早期worker文件，不能与原helper后写文件混淆。原生监督继续使用真实PID/FILETIME/exit与自有树清理。既有complete输出只可核验复用，不再执行；没有complete的部分输出不能自动按“续跑”重启。

共同环境/源码错误必须在条件循环前阻断。还须以一个小的明确保护覆盖循环内新出现的统一启动故障：在调用单条件worker的边界检查返回/异常诊断；若为确定的解释器、缺包、环境/源码/case身份错误，保存当前失败并终止整批，后续条件列not_attempted。旧循环捕获Exception，普通抛异常可能被吞掉并继续；实现必须合成验证真实停止传播，不能只在批末统计错误。允许wrapper安装局部适配回调而不改旧源文件，必须恢复回调，且不能以泛化字符串匹配把任何数值求解失败当环境失败。真实单条件数值/物理失败仍按原协议保留并继续独立条件；不补位、不重试。

实施QA范围限新接线：新旧目录拒绝、源/设计字节绑定、fit固定CAMPAIGN、FOM/GPU类别选择、wrapper早期worker、相同preflight生产调用、批级错误停止、六端点freeze路径、只核验complete不重跑。复用原9+4合成科学检查，不需要重跑整个环境probe或旧P0；必要新增QA及最多一次明确代码修复重验全计入本轮1000秒QA类别，不增加科学smoke。审查应核对这种进程内目录适配没有改变样本资格、源模型或冻结规则。

实际命令形式在实现后锁定为：FOM exe `-m experiments.first_physics_pair_20261020 acquire --stage short ...`，GPU exe同wrapper `acquire --stage score ...`、`fit adapt ...`。精确CLI由有限实现确定后写入命令JSON和协议，冻结并推送；本文不把尚未实现的接口称为已启动作业。原生包络与新launcher将类别、角色、固定路径、来源、remaining cap对应起来，不能仅通过argv看起来正确就宣布合格。

## 40000秒P3完整登记

工程闭账后父累计57943.4684632/259200，余额201256.5315368；P3余额88361.7061617，repair余额12174.8522211。**科学第一条启动前一次锁定40000秒P3**：父未分配161256.5315368，P3未分配48361.7061617。其他类别、旧900支持hold、原档关闭余额与原工程闭账保持。

| 类别 | 硬限进程秒 |
|---|---:|
| 新绑定准备与接线QA，最多一次允许修复重验 | 1000 |
| 48双步长短历史 | 1500 |
| D与入选六源15秒预测冻结 | 750 |
| 24双步长完整train FOM | 3000 |
| 六模型各256更新、加载、schedule与保存 | 18000，单模型所有attempt累计3000 |
| 适配后train评价、谱和导出 | 3125 |
| 24双步长完整val FOM | 3000 |
| 六模型val评价、freeze及统计 | 1875 |
| 固定支持 | 3250 |
| 非科学最终退出与归档 | 2000 |
| 原恢复重复预占保留为空额，禁止执行 | 2500 |

合计40000。恢复最多一次外部中断、≤31步重放，恢复实际≤2500且嵌套在该模型3000/六模型18000之内，上表末行不是额外恢复许可。类别不可互借。预flight、实际启动失败、监督、清理都计入对应外层；子进程不重复相加。结项以真实全部包络＋3250支持登记，明确取消未用预占回P3。任何退出超限如实计费，不用空额扩科学。

费用依据仍为原合法train24双步长640.484秒、旧短probe双步长约8.21秒和GPU完整12事件测时六模型追加256约12212秒上界推算；无需新cost smoke。不因环境修复改用小数据、少seed或少更新。每短条件双步长≤30秒；完整条件双步长≤120秒，每步55秒，保留退出余量。类别自检至少提前60秒停止新增任务，原生监督提前30秒，最终退出保留到cap。健康长作业按3–5分钟稀疏liveness和原登记科学里程碑监控。

## 科学阶段顺序及准入

1. root采纳窄政策、归档工程验收、登记40000、实现及有限审查、绑定QA通过；复制设计并锁定命令。24母线、候选每母线两层/每层16、val每母线1/每层8、随机12每层4及整数DP概率保持，不再声称每母线每层概率1/3。
2. 48候选各一次.005/.0025真实短FOM，first_record_plus_100ms，真实11点绝对时间/系统指纹一致、scaled RMS≤.001、双步长分类一致；unknown不能变stable，历史不足或数值失败不补位。保留48完整分母。
3. 对可初始化历史运行六源模型固定5秒未来、500个.01评分点、原21维量及D/B定义，ROM .005/.01。两个步长均D≥1且B≥2.5e-5、六源全合格才排序；排除预锁随机12，每层取4，按min(D)、min(B)、物理键固定排序。任一层不足4关闭本档，不能补随机、增候选或降门槛。
4. 任何完整FOM前冻结选定24×六源的15秒fine/coarse原预测及原始缺失/非有限标记。完整train双步长各一次，以真实系统/数值/共同11点历史对齐资格决定共同可用，不按ROM难度、分数或失稳标签排除；零可用关闭，部分合格可继续但完整24覆盖未决。原393来源/27排除/366合法train与新24分开。
5. 六源均从各自P2 global1024完整模型、Adam、Torch CPU/CUDA RNG继续，固定三seed6101/6102/6103和新采样流7101/7102/7103。同seed两结构共享冻结256schedule；原366＋共同新train，排序与覆盖轮询保持；相对1–256/global1025–1280，1/2/5时域86/85/85，LR .003/.0003/.00009、GPU float64及原损失/force保持。每32保存，128只看已存loss/grad/coverage，256唯一端点。不得把原1024训练调度直接延长替代新循环。
6. 六模型适配后原train与新train各评价一次；旧366未适配值引用P2，新train未适配只用完整FOM前冻结预测。分别报告1/2/5/8秒风险、共同频率rad/s、功率system pu/MW、失稳真实区间、分类/覆盖/所有数值失败。旧366分类前后共同8秒另表，不混15秒口径。导出六方程、GRU/尺度及39维Jacobian谱，不作L2证书。
7. 六个global1280端点全部完整，冻结六路径/hash、主seed6101、delete7−full8、设计与统计后才生成24val双步长；一次六模型评价，不替换失败，不看val后改配置/端点/seed。任何val不可覆盖或数值未决，全24确认未决。主指标仍为设计加权条件性稳定5秒风险（.8均值＋.2最高10%权重尾均值，边界分数权重），旧ceil未加权risk另列。逐事件配对及三seed方向保留，不新增95%显著性、bootstrap或伪重复。

只有全数值/覆盖有效且主seed改善、三seed同方向，才能描述固定设计下方向复现；频率/功率/失稳损害独立报告。局部Hurwitz、D高或工程通过都不能支持模型晋升或物理安全证书。

## 停止、归档与确定阻塞

目前**没有已证实的环境外部阻塞**：生产身份/spawn已经通过。科学启动的明确待完成条件是root窄政策采用、完整P3登记、新wrapper/launcher与fit目录绑定实现、有限QA/审查及源/命令冻结。fit.CAMPAIGN硬编码是已证实的接线限制，可按上述固定目录适配解决；它不是额外求解许可，也不是全任务阻塞。本文没有真实科学PID或已开始运行的声明。

新环境/源码确定性错误、泄漏、单位冲突或实现错误隔离受影响证据并ESCALATE；本档不以新编号重试。真实科学门槛不足按上述阶段关闭，不改标准。科学失败不能证明模型家族被否定；完整模型比较不足时保留独立有效结果。root依父授权检查后续合法方向，不因一档结束宣称任务完成。

root采用后应实际完成绑定与QA并启动合法的48短历史，不停在next_action。每执行阶段选择性commit/push并核实远端，先补齐任何待归档再新科学运行。工程可复用成果经范围有限PR审查才合并；模型采用仍需原科学门槛。默认模型、test关闭和TASK4暂停保持，旧失败、读取违规及P2的512数值未决永久保留。结项取得有界独立科学挑战，root记录最终科学解释与准确停止/恢复状态。
