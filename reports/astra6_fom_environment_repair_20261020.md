# Astra6：分离FOM环境修复与同一预锁设计的首次物理执行

2026-10-20；本次实际规划模型 **gpt-6-astra**，代理 `/root/astra6_after_fom_environment_failure`。这是本次有界只读规划的完整原始返回；root的采用决定应另行保存。本代理只读取协议、结果摘要和源码/环境静态元数据，未导入科研模块、读取轨迹或权重、运行FOM、训练或安装依赖；只新增本文件，未递归委派。报告应用anti-defensive-writing，必要的失败与权限边界完整保留。

建议先完成一个独立、无FOM的环境接线修复，再有条件登记完整科学阶段。环境修复应回答：**生产入口及其spawn子进程能否使用满足原系统身份的ANDES环境，而训练和推断继续使用原CUDA环境？** 科学问题保持：在相同预锁激励设计下，full8与delete7的预测取舍能否经新观测和共同适配复现。工程通过并不回答该科学问题。

## 已核实依据

权限依据为AGENTS.md、docs/engineering_loop.md及docs/current_state.md。原科学设计见reports/astra6_active_fom_20261020.md、reports/astra6_active_fom_20261020_design_amendment.md、reports/active_fom_pair_20261020_protocol.md；关闭依据为reports/active_fom_pair_20261020_conclusion.md、results/active_fom_pair_20261020/short_result.json和qa.json。

原48条件、96个步长调用均因缺少ANDES在物理求解开始前失败；0原始历史文件、0D、0完整train、0更新、0val。原生exit0表示失败清单完成，不能作为科学准入。9项采集QA和4组fit QA为合成检查，未验证生产FOM环境。211项绑定未变。原档实际62.2232031秒，加3250支持，共3312.2232031秒；未用36687.7767969已取消。root已核实负结果归档c3443ed7ddf68ed0b0ccc06e1db027e91c39f54d；main保持e126a01f34c45bdda1d1d0588bba93dd25e0a4fc，无本轮PR或模型采用。

静态源码核实：private_data/closure_mechanism_20261019/P3/active_fom_pair01/launch_stage.py将所有类别解释器固定为.venv-gpu。experiments/active_fom_pair_20261020.py顶层依赖NumPy、closure_p3_disagreement_20261020和closure_discovery_data；这些模块顶层没有torch导入，score内部才导入torch。closure_short_history_20261019同样没有顶层torch。closure_discovery_data._runtime_system_attestation实际导入ANDES、解析case路径、核对版本/case哈希/config/历史train指纹；FOM worker使用这一函数。不能仅因load_new_events处理新数据便断言它需要ANDES。

候选FOM解释器为 `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe`；训练解释器为 `E:/03-nn_Transfer_Func/kiki/.venv-gpu/Scripts/python.exe`。docs/gpu_training.md及实际pyvenv.cfg把前者明确列为GPU环境创建来源。两份pyvenv.cfg已只读核实，分别记录Python3.12.13/3.12.14；root静态核实前者存在andes及andes-2.0.0.dist-info、后者缺ANDES，configs/private_dataset.json要求2.0.0。**这些是候选环境证据，不是运行时合格证明，也不能断言它就是旧P0探针使用的解释器**；旧probe未记录exe。不得修改kron-reduce项目或它的环境，不安装包，不复制site-packages，不重跑旧P0探针。

## 第一段：独立有界工程修复，立即可以执行的范围

root采纳后登记 **repair类别1200进程秒：600实际包络硬限＋600固定支持**。包括接线、静态审查、一次工程QA以及至多一次明确实现修正后的重验；所有执行累计含在600内，不是每次600。支持计入一次；不用余额退出时取消。父累计由57325.6209421起算；登记后父未分配200674.3790579、repair未分配11592.6997422，P3保持88361.7061617。不借P0探针或原档已关闭QA余额。

新独立目录建议 `private_data/closure_mechanism_20261019/repair/fom_environment01`。保留原active_fom_pair01目录、原launcher、原source_manifest、全部attempt原样。实现限定为新私有launcher及小环境检查入口；必要的可复用修复可在新的受版本控制辅助模块中实现。不得修改训练损失、源模型、科学步长、物理配置或数据过滤。依赖源码身份与环境身份分开记录：外部解释器及ANDES/case不冒充本仓库source文件，不给原仅允许仓内source的verify_binding开放任意外部文件。

工程QA一次完整执行以下检查，使用合成请求与现有native监督，原始数据和模型权重不加载：

1. 两解释器均运行真实的轻量身份进程，保存规范绝对sys.executable、sys.prefix/base_prefix、Python版本、NumPy版本与模块路径。FOM环境真实import andes，保存版本/模块路径、andes.get_case解析路径及case哈希；读取明确许可配置及train系统指纹元数据，调用与生产相同的_runtime_system_attestation并通过其已有兼容性断言。可以import生产入口检查导入闭包，但不得调用求解、数据集加载或模型加载。若所用指纹函数含轨迹读取，改为此前已保存且绑定的train-only指纹元数据输入，不因此扩大读取范围。
2. FOM解释器通过Windows multiprocessing spawn产生一个只做上述身份/attestation的子进程；父子sys.executable必须一致并等于许可FOM exe。使用与真实FOM相同的spawn上下文和路径初始化，调用真实检查函数，不能以模拟字符串代替运行时验证。无andes.load、PFlow/TDS、simulate或任何轨迹生成。
3. GPU解释器真实import torch，记录版本、路径、CUDA版本/可用性和设备身份；不加载checkpoint、不做更新。确认score、schedule、adapt、evaluate、freeze/summary继续选择GPU环境；short、完整train FOM、val FOM选择FOM环境。prepare建议GPU环境以保留既有NumPy设计谱系；最终科学阶段直接复制冻结设计，无须重抽。
4. 合成命令测试覆盖类别与exe不匹配拒绝、FOM错误环境/缺包立即非零停止、子进程身份不一致拒绝、源码/环境/case身份变化拒绝、错误不落入48条件循环、native退出/自有树清理。用mock缺包做拒绝路径，不能为验证故障实际跑一遍48条件。每个科学FOM阶段在进入任何条件前执行同一真实环境preflight，生产worker仍保留原逐次runtime attestation。

建议新检查入口接口为 `environment_preflight.py --mode fom|gpu|spawn-qa --output <new-path>`；root实现后把**实际命令JSON和哈希**登记进工程协议。可执行形式为PowerShell `& 'E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe' <environment_preflight.py绝对路径> --mode fom --output <新输出路径>`，spawn-qa同一exe；gpu模式用 `E:/03-nn_Transfer_Func/kiki/.venv-gpu/Scripts/python.exe`。所有调用由已有原生监督包络执行，不能将本节尚未创建的路径称为已启动命令。检查入口不得接收科学condition或dataset参数。

环境身份不合格、版本/case不一致、生产导入失败、spawn不一致、600秒用完，均关闭工程段并记录具体阻塞；不能转而安装、换第三环境、重跑旧probe、以第二编号再次QA。一个可定位的代码接线错误允许上述一次修正重验。环境通过才进入下一段；工程验收为身份、拒绝路径与退出恢复证据，不要求产生物理历史。

## 第二段：条件复用的前瞻决定，不能隐含恢复原档

原文明确“科学FOM无失败重试”，条件设计又排除历史已尝试新train条件（含失败）。原档已关闭。这两条不能被解释成原本就允许再次调用48条件。**若root采用本建议，应明确登记一项只作用于未来阶段的窄政策：允许已证实没有进入物理求解、没有产生轨迹/标签/预测/评分的纯启动失败设计，在环境修复合格后执行一次首次真实物理实验。** 这是同一条件的再次软件启动、首次物理解算；不是“全新条件”，不是原档恢复或成功改判，也不是普遍的失败重试豁免。

此建议依据父任务持续研究授权、AGENTS.md允许独立新阶段并保留历史门槛、以及当前没有任何条件相关科学结果可供挑选。保留原48/24设计比因无关环境错误重抽更少引入研究者选择；没有依据要求换更容易的条件。原48失败在累计尝试中继续列出：旧48条件启动失败＋新阶段48条件首次物理尝试，唯一候选条件仍48；双步长启动计数也分别列明，绝不称96个独立条件。对于过去已进入物理求解或产生任何科学信息的失败条件，仍严格排除；不推广本例外。

采用的必要条件：root显式记录对原no-retry与失败排重条款的这一前瞻解释/新阶段政策；复核原96失败均在import ANDES处、raw数0、D/未来/val未产生；保存原design.json哈希c475a3bf11aaa17e2a3ca4642291a43d863541be14ec0fefdec5cd072de8c126，候选48、val24、随机12、DP概率、seed与顺序完全逐字沿用；原归档已推送；工程QA通过；新独立费用已锁定。建议在科学执行前取得一次有界只读挑战，专门检查该政策是否确实处于父授权、无条件相关信息泄漏且没有覆盖历史no-retry。由root作最终判断，不能以本文或新编号自行创设权限。

若root判断父授权不允许这种窄政策，则不得复用该48条件，也不得用“启动不算attempt”去改历史。此时先完成合法工程段；后续需实际Astra6重新登记排除原48和原val24后的新设计，并验证可行性，不能直接跑原设计。这是政策分支，不是在本计划内顺手多跑一个候选设计。

新完整阶段建议独立目录 `P3/active_fom_pair_first_physics01`，将旧设计按字节复制，另外写provenance关联旧失败档；不重抽、不改旧文件、不把旧环境失败记为新阶段物理失败。新source_manifest绑定修复后生产入口、新launcher、新协议及旧科学源码/源checkpoint的明确身份。必要的新source哈希与旧科学source谱系分别记录，不能让新解释器身份替换模型或物理源身份。所有源checkpoint保持原P2六个1024端点。

## 完整科学费用与顺序

新阶段仍锁定 **40000秒P3**，不缩六模型、三seed或256更新。它是从P3当前88361.7061617余额重新分配，非复活原40000；登记后P3未分配48361.7061617。工程段按实际加600支持闭账后，父剩余由实际余额减40000；按工程1200全额保守预占计，两段登记后父未分配160674.3790579。repair和P3不得互借；旧900 hold及其他类别保持。

沿用原类别硬限：科学接线/准备QA1000；48双步长短历史1500；D和15秒原预测冻结750；24双步长完整train3000；六模型256训练18000（每模型所有attempt累计3000）；train评价/谱/导出3125；24新val双步长3000；六模型val统计1875；支持3250；退出/归档2000。训练恢复2500为18000内嵌套上限，最多一次外部中断恢复、最多31步重放，不另加；重复保守预占的2500留空不执行。40000总cap和类别不可互借保持。工程段的环境QA不在P3重跑；P3的1000可用于新source/协议/设计绑定和必要合成接线验证，按实际计费。

费用仍有依据：旧合法train24双步长原生640.484秒；旧短probe两步3.763156/4.4471772秒；GPU完整12事件单更新上界原表推算六模型约12212秒，训练14400基础及18000硬限。改变启动解释器要求先证明物理环境身份合格，不要求另跑cost smoke；上述费用是完成机会的合理保守安排，不保证每条件成功。新阶段preflight和spawn开销计入对应包络；如费用超限，不缩seed或改科学门槛救济。

顺序和所有原科学门槛逐项沿用原协议及设计修订：复制绑定→48短历史→六源D、随机12＋主动12→24×六源15秒预测冻结→24完整train→共同池六模型各追加256→train评价与方程→六完整端点及主比较冻结→24新val→一次六模型评价→独立结项挑战→关闭。仍须真实11点、双步长RMS≤.001、unknown不变stable；主动每层不足4关闭，不加候选；train资格不按ROM误差删样本；六端点不齐不生成val；val任何未覆盖/数值未决保持全24确认未决。原test关闭，TASK4暂停，生成结构搜索关闭。

增加纯工程故障的批级停止规则：任何生产preflight失败，零条件启动且该段非零退出；条件worker如发现共同环境/源码身份错误，立即停止发出后续条件，保留已尝试和not_attempted身份，不让48次重复启动成为错误发现机制。真实单条件数值或物理资格失败按原协议保留、不重试、不补位，可继续独立已登记条件。再次确定性启动/实现失败关闭本次科学档，不允许再用本窄例外启动第三档；按累计预算和全任务合法分支重新决策。

## root执行与结项责任

root现在应保存采用决定和1200秒repair协议、实现上述小接线、登记实际原生命令并**执行一次环境QA**；不能只更新next_action。环境合格后保存其回执、实际成本和本报告，完成条件复用政策的root决定/有界挑战，再正式登记40000 P3新阶段并启动首次48物理采集。若需要实质修改本科学计划，再实际调用Astra6；本文已给出本次工程修复及有条件完整科学阶段的规划，不以heartbeat代替规划调用。

每次执行后按既有程序选择性归档并核实远端，原失败和旧档闭账永久保留。可复用接线若正确性/退出恢复检查通过，可单独经有限PR审查纳入main；完整科学结果不合格时模型和默认采用仍不变。局部环境修复、首次D或Hurwitz谱均不能替代L2或最终模型科学门槛。若停止，写明停在工程/政策/科学哪一段、真实作业是否退出、实际累计与未用取消额，并给出已有证据支持的下一动作；不把一个环境问题宣称为全任务资源耗尽。
