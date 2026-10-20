# 当前研究状态

任务状态：**RUNNING**，目录E:/03-nn_Transfer_Func/kiki。当前quadratic_memory_edge研究比较同源full8 global1280构造的A原项/B删除/C二次记忆项，三seed九组、统一新Adam，保持100ms一次GRU及1/2/5s循环。[协议](../reports/quadratic_memory_edge_20261020_protocol.md)、[实际gpt-6-astra方案](../reports/astra6_after_fresh_val_failure_20261020.md)、[独立实现审查](../reports/quadratic_memory_edge_20261020_implementation_review.md)。

**实际执行**：九组512训练、九次固定端点评价全部完成，无重试/恢复/768。真实worker均退出0且cleanupGONE，来源及逐行训练日程一致，九评价各385完整事件，预测与39维局部结果保留。九端点manifest SHA de2f3434a2b2bd1aaed0e09a100cafb53b4b3049914bfbe5f5affb59c2360a39。当前无科学进程，下一实际动作是按已登记JSON-only命令统一提取原366/新19/共同385指标，不重做预测、不用runner错误prefix分栏。指标提取后root决定，默认不变。

**费用**：父上限259200、旧关闭70318.2526126；本轮支持hold3000+工程253.7854629+训练12158.9154373+评价967.1775328，累计86698.1310456、剩余172501.8689544。本轮P2预占36000，尚未关闭、未用19620.1215670，父未分配152881.7473874/P2未分配30100.6537796。工程2500/训练27000/评价3500/支持3000不可互借；旧900支持hold保留。[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**边界与基线**：共同385=原366+新19（342稳定/43失稳），新5失败保留。fresh_design_pair已关闭并归档8aa736ce2104079d6931d11244248c69beb5f31d；六模型1280与18次train评价范围见[结论](../reports/fresh_design_pair_20261020_conclusion.md)。其val路径接口在求解前拒绝，0物理条件/0轨迹/0成绩；旧24不重试、不搬模型、不放宽guard。当前不访问FOM/val/test，TASK4暂停、生成结构搜索关闭、默认模型不变。旧18组GRU512耗6085.6472446不重跑。

**恢复与归档**：BASE=private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01；固定源码ee922ca3e0640768a8d1dea5d4e3f8bb4382a1a6，源full8检查点在P3/fresh_design_pair01/models/{seed}_full8/checkpoints/step001280。已完成任务退出0且cleanupGONE，实际命令/绑定/回执在BASE；源与全部Adam保留。root归档codex/quadratic-memory-edge-20261020，执行代理独占codex/quadratic-memory-runs-20261020，九评价末项事实远端81f48ffdcb8bafdccb52a67a9ccfa95de0b515ae。QA01失败保留，唯一guard修复后QA02 27项全部通过。工程PR15已合并并核对main c2c95a75f455e2b3e426ca98dd5fe06c93b327a4，仅可选runtime/测试/说明三文件；工程接受不等于模型采用。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb/index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93保持，用户工作未重置。heartbeat ACTIVE，健康无变化保持安静。
