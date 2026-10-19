# 当前研究状态

任务状态：**RUNNING：闭合机制研究持续推进**。工作目录E:/03-nn_Transfer_Func/kiki；原test关闭，TASK4暂停，默认模型不变，新模型均为研究候选。

**当前唯一作业：P0固定边项完整接口与费用测量，已实际启动。** 2026-10-19 16:44:30本地启动；实际计算PID67828，计算launcher60876，监督进程82708，顶层launcher52620。已核对进程命令与父子关系，无重复队列。输出private_data/closure_mechanism_20261019/P0/edge_pair_gate01；progress.json已进入GPU试算更新，真实366train、面板和3×1024采样身份已先冻结，frozen_identity.json SHA256 8c3302dc0e810c1c42df2f6a54f3f430ea14ae870a2cfea7c850453e8a9ff263。

本轮由实际gpt-6-astra /root/astra6_after_saturation_route_closed规划，root采纳[有限协议](../reports/astra6_edge_pair_gate_20261019.md)，源码/QA已推送codex/p0-edge-pair-gate-20261019，提交3ea5961fd7b62041b774d76305f4e6f2516a41c3。新driver SHA256 5a15cf58e5c3e56ce9e38a01e59ca7e06ea9e49eeb0256cca64f13ca42433fcb。旧数学和数据源码不改；从登记S-O重建原八项（不含port es），只删除整项edge (qi+qj)c得到七项。公开包仅提供terms，不采用其训练权重。

两次实际CUDA检查均通过，累计QA46.1312582秒（计800支持内，不双加），一次修正用于Windows自有进程树终止，检查次数已用尽。场/能量梯度、完整目标/全部梯度、Adam、导出、恢复及下一更新与参考一致，独立只读审查接受限定工程检查。见[紧凑QA证据](../results/edge_pair_gate_20261019/qa_summary.json)。

实际命令：`.venv-gpu/Scripts/python.exe experiments/edge_pair_cost_gate_20261019.py measure --output private_data/closure_mechanism_20261019/P0/edge_pair_gate01 --max-seconds 2800`。EXEC只有一批：两结构×三时域×三train面板×两更新=36试算，权重丢弃；两来源结构重新构造后各全366train冻结评价/数值资格、局部分析与保存恢复测量。总2800秒含启动/子进程只记一次，超时终止自有进程树。不得重跑、续跑本目录或追加第二批。下一动作稀疏核对最终receipt.json、measurement.json及必要失败日志，核实所有子进程后评价/关闭/归档；关键测量缺失保持未决。费用通过也不授权P2，后者还需实际Astra6协议及原授权范围独立挑战。

预算：父259200，P0/P1/P2上限14400/28800/86400保持，未跨阶段转移。已关闭P0可核实际142.1922393及83.2287491秒，旧未核900支持与饱和工程600支持保守保留，原P0未占12674.5790116。本轮前瞻3600=EXEC2800+QA/审查/归档支持800，分配后P0余9074.5790116，父余229056.6693887。当前EXEC尚未结账，不把上限写成实际消耗。P1实际加保守支持累计24817.9096229、余3982.0903771；P2未启动且86400未用。见[费用快照](../results/closure_mechanism_20261019/cost_snapshot.json)。旧账本不转移，支持预占不能随意转成训练。

P1保留已知支撑、无噪声节点/边的预测证据：各三种子×两初始化器×512，候选合成val总体MSE均3/3≤0.01、配对对照0/3，全部模型数值合格。节点候选0.00148505/0.00228073/0.00234259，边0.00082521/0.00220175/0.00123593；不是未知结构发现、真实q唯一性、独立确认或稳定域证书。原18项GRU已知支撑512均完成、成本6085.6472446，不重跑；原q0诊断按4500秒关闭，不恢复。

饱和比较实现路线已CLOSED：run01 PID79820退出，六项零更新身份失败、无检查点或模型val成绩；唯一修复QA2未通过不可变buffer身份条件，run02未启动，不再诊断/修补/检查。实际22.791974+800支持保守保留，未消费3977.208026明确撤销。负记录和源在[codex/saturation-pair归档](https://github.com/LLzijian/grid-transient-rom/blob/3792653e0e459ac0e6a3c94b0ef747dfd16f7728/reports/saturation_pair_20261019_conclusion.md)，不合main，不改变节点/边结论。先前专用tanh固定工程网络检查/测时有效，但通用采用暂缓，见[工程结论](https://github.com/LLzijian/grid-transient-rom/blob/a214680ba4827c3099b9e8cc96261c6622e40f45/reports/tanh_port_runtime_20261019_conclusion.md)。本P0不依赖或重开该修复。

P1饱和/未知恢复条件未完成，发现门槛未通过，真实生成搜索关闭。当前P0仅核验原方案允许的“因果恢复＋固定边项存在性”收缩研究是否可执行，不把P1改称通过，不提前启动P2/新val/P4。边结果与此前P0方案已通过PR #7合main d566eb272d73d9507c4f2e6d58b58720e7be1930；当前轮未合main。原本地HEAD a951013及Git index/用户未提交工作保持，不reset、不强推、不删证据。
