# 未来原生监督回执：一次有限工程修复方案

规划模型：gpt-6-astra（Astra6）；代理：/root/astra6_native_receipt_recovery；日期：2026-10-20。本文件是该代理完整原始方案。root须另存采纳决定和执行登记；本方案未启动任何QA、推断、训练或FOM。此次工作只读文档、源代码和既有合成QA回执，唯一写入为本文件。

## 建议与作用

建议采纳一个repair分支：将未来原生进程监督收敛为一个很小的显式Process生命周期接口，并用快速退出0、快速退出7、超时清理三项固定合成案例验收一次。它直接解决下一真实实验需要可信退出状态与费用包络的问题，有明确工程价值。它不能补写历史退出码，也不恢复原P3协议的执行资格。

本轮只有一个实现、一个正式QA包络、零重试、零扩展。通过则可以将被验收的未来接口送有限工程PR；未通过则关闭这一实现分支，不再追加同类监督修复轮。原P3分歧分支在本轮结束前始终关闭。本轮不进行额外科学诊断，不读取任何数据集、模型或检查点。

## 直接依据和判断边界

- 权限与执行规则：AGENTS.md、docs/current_state.md、docs/engineering_loop.md。
- 原P3登记与未决结论：reports/p3_disagreement_20261020_protocol.md、reports/p3_disagreement_20261020_conclusion.md。
- 读取违规：reports/p3_disagreement_20261020_preparation_incident.md。首次整HDF5哈希读到了受限原始字节；未反序列化或评分。原失败源码与manifest永久保留，不能改称test未读取。
- 故障监督源：private_data/closure_mechanism_20261019/P3/disagreement01/native_supervise.ps1。它使用Start-Process -PassThru，WaitForExit有界等待，finally先Refresh再读HasExited/ExitCode；native_receipt_complete目前只表达HasExited，未要求退出码存在。
- 成功对照：reports/fixed_edge_worker_resume_20261020_conclusion.md、results/worker_resume_20261020/summary.json及private_data/closure_mechanism_20261019/repair/worker_resume01/qa01.native.receipt.json。先前相同监督写法在57.5088199秒QA回执中获得exit0。旧QA内部的正常、非零7、超时清理也有成功记录；这不能自动验收新PowerShell接口。

已知事实是原P3合成7项PASS而原生exit_code=null、status=FAILED、完整包络3.4262628秒。不能据内部OK填0。进程对象或句柄生命周期、快速退出与Refresh之间的相互作用只是候选解释；静态材料不足以确定Windows PowerShell缺陷或具体竞态。无需先重复旧监督来证明根因，未来接口按明确契约进行一次验收即可。通过支持新实现的限定能力，不支持“历史根因已证实”。

## 唯一未来接口

建议新增experiments/native_process_receipt_20261020.ps1，仅负责一个有界子进程；不导入任何科学模块，不配置调度、恢复队列或通用作业平台。接口参数限于可执行文件绝对路径、固定参数、工作目录、独占输出目录、执行上限及来源commit。命令、参数、实际PowerShell版本、Python路径等全部写入本轮记录。PowerShell主机固定为历史调用使用的Windows PowerShell，并记录其实际版本，不能在测试中无声改用另一主机。

实现要求：

1. 显式创建System.Diagnostics.ProcessStartInfo和Process；UseShellExecute=false、CreateNoWindow=true，工作目录明确。以同一Process实例Start、等待、读退出码和Dispose。Start成功后立即取得并保留本实例的原生Handle，直至退出码读取及回执写入完成。不要Refresh、Close或重新按PID取得一个Process来代替原实例。
2. stdout/stderr若重定向，必须在有界等待前同时启动异步读取，避免串行ReadToEnd堵塞。等待、日志排空、清理和回执发布均受包络截止时间约束；不能靠无界WaitForExit或无界任务等待取得“完整”状态。固定验收输出很小，不需要建立日志服务。
3. 按截止时间显式等待。正常退出后，在仍保有原实例/句柄时立刻读取ExitCode并保存为可空整数；读取异常或缺失必须保留null与错误。不能将null强转成0；不能用日志文字、worker.json或测试期望替代OS返回值。
4. 超时先标记timeout=true，再清理该次创建且身份可核对的自有进程树，完成有界退出等待。记录taskkill返回状态、被监督PID、测试子PID及存活核查。不得按名称清理其他Python、PowerShell或用户进程。仅用PID检查时要配合创建时间/已有句柄，避免PID复用误判。
5. Process创建失败、等待失败、清理失败、日志排空失败和退出码读取失败均生成明确失败回执。回执保存已观察状态而不推断成功；日志排空与清理异常不能吞掉原错误。输出目录预先排他占用，存在即拒绝，历史路径不可覆盖。
6. 回执区分“进程已终止”“退出码已取得”“清理已核实”和“任务成功”。native_receipt_complete至少要求终止已观察、退出码为整数、所需日志及清理完成、费用信息可追溯。非零正常退出可以是完整失败回执；超时即使取得退出码0也必须保持TIMEOUT，绝不COMPLETED。完整合格成功要求未超时、exit0、无监督错误及包络在上限内。
7. Stopwatch从启动尝试前计到子进程终止、清理、日志排空与退出码取得后；回执写入/监督退出的开销由外层验收包络覆盖。只有已记录字段可以宣称完整。Dispose放在回执写入之后。保留真实exit7及超时实际取得的OS退出码，不规定强杀后的特定整数。

如果实现发现上述最小接口无法在有限范围内可靠完成，不另建平台，不扩大权限；静态审查不准入或唯一QA失败即按下面的关闭规则处理。

## 唯一合成验收

建议新增tests/test_native_process_receipt_20261020.py，使用标准库编排真实Windows PowerShell进程及上述同一未来接口；不模拟Process、WaitForExit或ExitCode。测试材料在新目录private_data/closure_mechanism_20261019/repair/native_receipt01/qa01。外层标准库Popen实际等待监督进程，记录其OS退出码和完整包络；不再使用本次故障旧脚本包住新验收，以免重复同一未决依赖。外层也须有真实截止时间和自有子树清理。只可运行这一套一次，按固定顺序逐项一次，无循环抽样、预跑、重试或失败后换实现再测。

| 固定案例 | 进程行为 | 必须观察的验收结果 |
|---|---|---|
| quick_exit_0 | 固定本地Python环境执行纯标准库fixture，写少量stdout/stderr及PID记录后立即exit0，不加人为sleep | launcher真实exit0；完整回执COMPLETED；timeout=false；日志内容符合fixture；监督OS退出为0；无遗留自有进程 |
| quick_exit_7 | 相同fixture立即exit7 | launcher真实exit7原样保留；完整回执FAILED_NONZERO或等价已登记状态；timeout=false；监督OS退出非0；外层QA正确接受此预期失败 |
| timeout_tree | fixture创建一个已登记子进程，两者保持存活，预先写入PID/创建时间；不接触数据；运行限10秒，留至多10秒清理 | timeout=true；状态TIMEOUT；实际父子均终止，日志/回执保留；退出码确由OS取得；监督非0；无遗留自有进程 |

快速案例每项包络上限30秒，包含清理与日志。超时案例整个包络上限30秒。三案例固定顺序只执行一遍，总正式QA上限300秒，余量给主机启动和完整外层退出。出现首个非预期失败则停止余下案例并关闭QA；不能用后续重试补齐。三个预期行为全部满足、外层原生exit0且完整包络不超过300秒，才是ENGINEERING_ACCEPTED。仅JSON PASS、仅日志OK或仅子案例完成均不够。

测试命令由root在冻结前落实为tests/test_native_process_receipt_20261020.py的唯一命令（--output上述新qa01、--max-seconds 300），明确绝对Python和powershell.exe路径。记录真实launcher/worker身份；环境Python若带重定向launcher，回执应区分launcher与实际fixture PID，不能混写。监督退出码代表它实际启动的子进程，fixture PID记录用于验证作用对象和清理，不用作退出码替代。

静态审查可检查源码、参数转义、固定案例、异常分支和数据访问范围，但不得以“烟测”“导入检查”“手工验证”再启动案例。root需要一次有界独立只读挑战，覆盖这个实际待执行接口及验收来源；审查前不运行。该审查也可承担最终工程PR审查，避免无必要重复。

## 费用与文件边界

新登记repair总额900秒：唯一QA300秒，支持保守600秒，类别内部两项不互借。只从已验证repair余额13407.9059650中分配，登记后repair可用12507.9059650。父累计52375.5471438，父剩余206824.4528562；登记后父未分配205924.4528562。结项费用为真实完整QA包络加600支持；若未启动QA，也记录已经发生的支持成本，不记零。未使用QA余额明确取消并回repair类别，不移至P3。实际意外超限仍如实累计并关闭，不把上限当作实耗。旧900支持hold保持。

本次不动P0=9037.2289973、P1=3982.0903771、P2=66100.6537796、P3=92696.5737372、P4=21600各余额。原P3登记4800已经结项903.4262628、余3896.5737372已取消；新900不是续用其未执行3600，也不恢复其QA次数。

除上述两个未来源码文件，执行阶段允许新增短protocol、conclusion、results/native_receipt_20261020/summary.json及必要本地合成日志/回执；root按真实状态更新docs/current_state.md与现有cost_snapshot.json。不得修改旧native_supervise.ps1、旧qa01、冻结manifest、科学runner或模型。来源冻结仅针对明确的源码/配置文件，不递归扫描private_data，不读取任何HDF5或模型权重，不导入科学模块。没有训练、D、Jacobian、FOM、train/val/test数据访问。

## 关闭、集成与下一决定

本轮成功仅支持该主机、该实现和三个固定案例所覆盖的未来监督能力。可提交范围受限工程PR，审查精确diff及证据，核对预期head、远端merge；没有模型采用、结构发现或稳定性结论。原PR11 main51370332dd68054028fdeeb86d967c8cc5b72ffb及原六科学端点分类保持。工程验收未通过则不合并新接口，不改默认入口，保留全部源码与不利证据。

无论成败均由root选择性归档协议、源、回执摘要、实际命令和费用，并核实远端提交。先前归档d874c8ae96b280590facc0e2306dbcc0bd2ea3fd及原exitnull记录保留；不能把新成功写回原P3为exit0或已通过。

这是一次具体阻断的终端修复机会，不是开放式repair链。成功后的下一次实际Astra6规划只能审议尚未执行的固定D是否在原科学定义及新独立授权预算内值得执行；须明确旧P3仍关闭、旧唯一QA不可重跑，不能靠同名改目录续跑。D的问题、epsilon、门槛、六端点与事件分母不能据此改变，本轮也不预先授权D。若合成修复失败，则关闭本实现分支与依赖该监督的P3执行路径；下一轮不能再换监督实现或堆诊断救该支路，须在已授权其他路径中选择独立问题，或记录具体阻塞并保留安全准备。

工程修复不属于以指标代替自主模型证据的proxy诊断，故不凭名称重置原两次诊断限制，也不占用机会后宣称科学进展。P1未知结构搜索仍关闭；P2删除边项的两好一坏及频率/失稳样本损害保持；P4原准入仍未获得。尚未执行D不是D失败，更不是全局任务完成或阻塞的证据。最终下一阶段与科学结论由root负责，不能由本方案自行授予权限。

root需明确采纳的五项：900秒repair独立分配；一个Process实例/持有句柄的未来接口；三案例唯一QA且零重试；原P3及读取违规记录不变；成功只进入工程PR和另轮D审议，失败终止本修复分支。任一项不能成立则不启动本轮QA。
