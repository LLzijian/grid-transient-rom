# 当前研究状态

任务状态：**RUNNING**，父研究继续，工作目录E:/03-nn_Transfer_Func/kiki。fresh_design_pair已关闭：train研究完成，val因路径接口不一致在求解前拒绝，0物理条件/0轨迹/0成绩；本轮不集成、不重试、不采用模型，当前无科学PID。[结论](../reports/fresh_design_pair_20261020_conclusion.md)、[失败回执](../results/fresh_design_pair_20261020/val_start_failure.json)、[闭合账本](../results/fresh_design_pair_20261020/closed_cost.json)。实际gpt-6-astra已返回二次边记忆固定函数方案，root按独立挑战采纳三同源结构、九组Adam统一重启，保留1/2/5s循环；登记P2 36000秒（工程2500/训练27000/评价3500/支持3000），尚未启动科学训练。当前执行：QA01实际26通过/1失败（未缓存field结构验证遗漏），14.1351105秒且270源前后一致/worker清理GONE；使用唯一实现修复后复测，尚未正式训练。专用快路径、同源构造及成本准入；协议见[quadratic_memory_edge](../reports/quadratic_memory_edge_20261020_protocol.md)。旧24禁止重试。

**保留进展**：48短FOM中41合格，固定D/B准入37；24完整train尝试19合格（17稳定/2失稳）、5失败或未决全部保留。共同385=原366+新19，六组GPU从1024连同Adam/RNG联合适配256至1280，均全385覆盖，无重试/恢复；18次固定train评价闭合。新17稳定train完整5秒风险full8三seed3.433179/3.495122/3.723859，delete7为1.099224/1.041328/.860202，所有新19队列数值合格；新两失稳样本实际区间误差删除组更大且六模型均漏检。train不是独立确认，也不能隔离新增数据的因果收益。

**数值与确认边界**：原366的6101/6102 delete7之expansion_0094未过8秒资格，primary=null；6101的15秒通过不恢复8秒、6102的15秒也未过。原366 full8风险2.472864/1.854441/2.252939，delete7未决/未决/1.740672。三delete7局部39维线性化Hurwitz，full8各一个不稳定模态，均无L2区域证书。六端点及统计已冻结，但旧verify_val_freeze要求模型在campaign01内，而合法新模型位于BASE/models，故val拒绝。原test关闭、TASK4暂停、生成结构搜索关闭、默认模型不变。

**费用与资源**：父上限259200，累计已关闭实际加保守支持70318.2526126，余188881.7473874。fresh档33份回执原生5871.1376375+3250支持=9121.1376375，未用30878.8623625取消回P3。各类余P0 9037.2289973/P1 3982.0903771/P2 66100.6537796/P3 75986.9220123/P4 21600/repair 12174.8522211；当前P2新轮预占36000，父未分配152881.7473874、P2未分配30100.6537796，类别不可互借，旧900未核对支持hold保留，子进程不双计。[累计账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**恢复与归档**：BASE=private_data/closure_mechanism_20261019/P3/fresh_design_pair01永久保留，来源4ae86576e4cdae1fdca912b0f36172331ee60f10；val session86911已exit1、worker70068/FILETIME134369253714201900已GONE，967源未变。冻结SHA7b6bca11eed6e0fb2da63f908d3d78728f3962ff3dc719c9a218c827fc5005cb不改；六预备val评价未执行，不是作业队列。所有源码、六模型、Adam/RNG、事前预测及失败记录保留，私有资产路径/hash在结果中。本机17项接线QA需要私有launcher/环境QA/条件metadata，不声称公开clone自足CI。

main保留6b17be3535cc57985c1e1be4138decabc7c0f033（工程PR14）；fresh证据选择性同步codex/fresh-design-pair-20261020，不直接整体合并。旧失败档c3443ed7ddf68ed0b0ccc06e1db027e91c39f54d、ef0aecfb18a28d79345510fcb6b040c05ea466c5及受限字节读取历史违规保留。原18组GRU512已完成6085.6472446秒不重跑。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb、index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93和用户修改保持，不reset/强推/删证据。heartbeat ACTIVE，健康无变化保持安静。
