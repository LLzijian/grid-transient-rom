# 当前研究状态

任务状态：**RUNNING**。工作目录E:/03-nn_Transfer_Func/kiki。当前fresh_design_pair正在执行固定结构新激励研究：[实际Astra6原案](../reports/astra6_after_first_physics_QA_20261020.md)、[root协议](../reports/fresh_design_pair_20261020_protocol.md)、[独立审查](../reports/fresh_design_pair_20261020_independent_review.md)。原test关闭，TASK4暂停，真实结构生成搜索关闭，默认模型不变。

**当前结果**：48个短FOM条件41合格、固定D/B准入37；24个完整train条件全部尝试，19合格（17稳定、2失稳），其余5完整保留、不补抽。完整24覆盖仍未决。共同训练池385=原366+新19，三采样流7101/2/3各256更新，1/2/5秒86/85/85，同seed两结构共用日程，SHA5cfda5e955ce165cf342a31ea40a547a700a02e838b8766ed9ed17f12397227f。

**当前作业/下一动作**：六组GPU适配均已完成global1280、各256更新/全385覆盖，原生训练3893.9836651秒，无重试/恢复；[六端点](../results/fresh_design_pair_20261020/adaptation_complete.json)。固定18次train评价已闭合3项，当前6101_delete7/original_after实际worker20640/FILETIME134369241298208740、监督86668/PS66580、session63325，native_eval_train_6101_delete7_original_after，勿重复队列。评价CPU float64单线程：原366适配后自主、新19适配后自主、新19源1024事前预测离线分开；原366未适配只引用P2。原366native360/self300、新19两类native180/self120，累计实际116.5432332/3125秒，以实际加下一cap门控，无重试或类别借用；[前瞻包络决定](../results/fresh_design_pair_20261020/evaluation_execution_decision.json)。仍保留新train全部24尝试及5未合格分母；全部评价/统计及六端点冻结后才生成新val。BASE=private_data/closure_mechanism_20261019/P3/fresh_design_pair01，来源4ae86576e4cdae1fdca912b0f36172331ee60f10，评价命令归档d9e87166cae6f63a94e139ec2eb28206636a6ea3。无模型采用。

**预算**：父上限259200进程秒，已关闭轮累计61197.1149751，余198002.8850249；当前40000秒P3预占单列，已结束原生5294.5842903（含适配前1284.0573920），支持3250保守预占。训练类别已用3904.7362489（含schedule10.7525838），上限18000；后续逐组核对实际费用及包络。父未分配158002.8850249，P3未分配45108.0596498；类别不可互借，旧900未核对hold保留，子进程不双计。[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。未分配前各类余额P0 9037.2289973/P1 3982.0903771/P2 66100.6537796/P3 85108.0596498/P4 21600/repair 12174.8522211。

**既有证据**：P2六组1024[固定结构结果](../results/fixed_edge_pair_20261019/final.json)两seed删除组风险改善、一seed损害，删除组三seed共同频率与失稳误差损害；局部谱不构成L2证书。P1未知结构恢复未通过；18组GRU512已花6085.6472446秒，不重跑。首次真实D已完成并合并PR13。原active_fom因缺ANDES在物理求解前失败、首次物理档两次接线QA失败，均永久保留；旧设计复用例外关闭，当前使用排除旧72后的384池。此前受限字节读取[历史违规](../reports/p3_disagreement_20261020_preparation_incident.md)保留，不追改历史。

**Git与恢复**：main已核实6b17be3535cc57985c1e1be4138decabc7c0f033（工程PR14），本轮暂存于codex/fresh-design-pair-20261020；工程合并不等于模型晋升。源检查点/原始轨迹/中间检查点本地保留，重要文件路径/hash在结果中，复现需要私有资产。原失败档归档c3443ed7ddf68ed0b0ccc06e1db027e91c39f54d、ef0aecfb18a28d79345510fcb6b040c05ea466c5。本地HEAD a95101337855ed3fc1f4abfd710982443d5564cb、index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93及用户修改保持。heartbeat ACTIVE；健康作业稀疏监控，完成/关键失败或可信变化才通知。
