# 当前研究状态

任务状态：**RUNNING**。工作目录E:/03-nn_Transfer_Func/kiki。新条件完整档已关闭：48候选的96短历史调用全部在FOM启动前因GPU环境缺ANDES失败，0合格历史、0D、0完整FOM、0更新、0val。真实worker42944已退出，内外exit0只代表失败记录完成，清理/211绑定文件核实；无当前科学PID。原test关闭、TASK4暂停、默认模型和main不变，本轮集成未采用。

**本轮入口**：[root有限协议](../reports/active_fom_pair_20261020_protocol.md)、[实际Astra6原案](../reports/astra6_active_fom_20261020.md)、[执行前原始修订](../reports/astra6_active_fom_20261020_design_amendment.md)、[独立挑战](../reports/active_fom_20261020_independent_precheck.md)。实际模型gpt-6-astra已调用，root采纳。完整档48双步长短历史、随机12+主动12的24新train、两结构三seed各追加256、六端点冻结后24新val一次确认。旧val只读事先条件设计排重；可用456组合，1-short与3-long为空，采用固定整数DP平衡分配。原参数域不扩，不再声称每母线每层概率1/3。本档科学尝试1（全部在FOM启动前失败），无合格科学结果。

**下一实际动作**：[环境工程结论](../reports/fom_environment_repair_20261020_conclusion.md)已实际PASS：真实ANDES/生产attestation/spawn/GPU与9guard通过，内外exit0、清理GONE、90source未变。唯一QA17.8475211秒，加600支持结项617.8475211，取消582.1524789回repair；无当前PID。root接受工程，正在有限PR审查；实际gpt-6-astra已再次调用，编写独立首次物理执行方案。该科学阶段尚未登记或启动；原48失败保持，test关闭、默认模型不变。

**预算**：父上限259200进程秒；已关闭轮累计实际加保守支持57943.4684632，余201256.5315368。P0余9037.2289973、P1余3982.0903771、P2余66100.6537796、P3余88361.7061617、P4余21600、repair余12174.8522211。新条件档实际QA13.956230+准备7.6523984+短阶段40.6145747=62.2232031，加3250保守支持，结项3312.2232031；取消未用36687.7767969回P3，无在途预占。类别不可互借，旧900未核对hold保留，子进程不双计。见[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**决定性既有证据**：首次真实D[结论](../reports/first_disagreement_inference_20261020_conclusion.md)/[摘要](../results/first_disagreement_inference_20261020/summary.json)/[事件表](../results/first_disagreement_inference_20261020/events.json)已关闭并合并PR13。六模型366事件2196项双步长核验合格，361事件/24母线满足固定D/B判据，fine D中位数13.32821081；只支持采集设计准入。唯一实际外层122.6443724秒、内外exit0、清理与70源核实；结项1022.6443724含900支持，未用3477.3556276已取消。private_data/closure_mechanism_20261019/P3/first_disagreement_inference01完成，禁止重跑/恢复。

P2六模型各1024已关闭，[结论](../reports/fixed_edge_pair_20261019_conclusion.md)/[最终证据](../results/fixed_edge_pair_20261019/final.json)，来源47ba6f341886aeb4cc235d3c25a9b16bc8f4a24b。325稳定5秒风险完整/删除：6101为2.695252/1.847689，6102为2.215011/2.725425，6103为2.747314/1.941173；两改善一损害。删除组三seed共同频率与失稳样本误差损害；删除39维局部谱Hurwitz、完整各1不稳定模态，不构成非线性区域或FOM安全证书。6102_delete7在512时一事件数值未决永久保留。结项20299.3462204，未用51700.6537796取消回P2；目录P2/fixed_edge_pair01/campaign01不得重复。

**保留的历史边界**：P1未知结构恢复未通过，真实生成搜索关闭；原18组GRU512已完成6085.6472446秒，不重跑。P0短历史双步长真提前停止probe已合格，首记录+100ms，实测切除后预测起点约0.1001秒；不等于完整事件资格。原P3的exitnull未决和首次准备整文件hash读取历史混合HDF5受限字节的[违规记录](../reports/p3_disagreement_20261020_preparation_incident.md)保留；未反序列化/评分不能抹去读取事实。后续成功不追改历史。

**Git与恢复**：本轮协议已推送核实codex/active-fom-pair-20261020的15274fa781be64c48127fe4491f40453889a918b；本轮关闭未集成，无PR或模型采用。最近已合并工程PR8–13，核实main e126a01f34c45bdda1d1d0588bba93dd25e0a4fc；原生回执PR12与未来恢复PR11已通过对应工程检查。每执行阶段选择性归档，有可验证工程进展且审查通过才PR合并；所有负结果保留。私有原始数据/权重仅本地，重要文件记录路径/hash。本地HEAD a95101337855ed3fc1f4abfd710982443d5564cb、index SHA256 82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93及用户未提交工作保持；不reset、强推或删证据。30分钟heartbeat ACTIVE，健康作业稀疏检查，关键变化才通知。
