# 当前研究状态

任务状态：**RUNNING**，E:/03-nn_Transfer_Func/kiki。当前两种边记忆方程有限辨别：[协议](../reports/edge_memory_discrimination_20261021_protocol.md)/[P1登记](../reports/edge_memory_discrimination_20261021_P1_registration.md)。实际gpt-6-astra原案addendum2及独立方案/实现挑战已采纳；只比较正弦差与余弦和两项，不称185项发现、不放行真实广搜。

**实际状态**：P0唯一QA11通过，3旧对照stage0/RNG/空Adam全部一致；native44.576883秒、101源/两层0/GONE。唯一cost12次试算丢弃，通过：native17.5794005秒，103源/97科学源与QA一致、两层0/GONE；含25%余量训练333.3194/229.0089/226.6731<=640、评价27.4793/27.3082/27.3675<=40。P0关闭，不重测/修源。P1已登记3300，正在冻结新启动文件，**无活原生作业/PID**；下一动作依次三个新cos各512→统一冻结train选择→各一次16train/16合成val双步长评价。旧三个sin512与指标冻结复用，不重训/重评/热启动。QA源219faa74a63aa9090f8f5a9148ebe0a57f6d4aa4，runner SHA93a579962ae6383df292913a0ff6a81d2a31a9c44d40e8dc5115860b214275f8不变。

**预算**：父上限259200。P0关闭930.5387226000223/2200（static150保守整帽、support700、meta8.3824391+closing10、native62.1562835），未花1269.4612774取消；P0余7192.2323664。新P1总3300=3train700+3eval100+metadata100+support800，P1未分配682.0903771；metadata100含prep40/freeze50/closing10，不重计freeze。父新P1登记时support800后累计136003.0084385，剩余未执行最大2500另留，父未分配120696.9915615。P2余3060.8945846已扣本规划600，P3余75986.9220123/P4 21600/repair12174.8522211不互借，旧900/P8尾/W尾40等保留。[账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**已关闭W8**：三1024/385评价已完整关闭，共同稳定5s风险降10.36%–11.21%但共同频率误差升2.29%–2.57%、漏检未改善，新19稳定风险全升；原门槛FAIL不采用/默认/确认/P4。合格runtime经[PR23](https://github.com/LLzijian/grid-transient-rom/pull/23)精确3文件合main27aa39ad2a2821adefbee1913377a9af153c868c，merged/main/3blob已验证；科学证据0ca18dad79fa5abf1f66faf6e97c1e29acaa9558。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb/index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93未改。

**权限/恢复**：真实train385=原366+新19，342稳定43失稳；40状态/100ms一次GRU/真实区间/单位/数值标准保持。新试验仅既有合成边train/val，项目val/test关闭、TASK4暂停。原18×512的6085.6472446、q0、node/edge旧作业、饱和及E/P8/D/S/F关闭不复活；P1广搜门槛仍未通过，P3配额用尽/旧24valguard不重试、不搬模型，P4同候选L1未满足。父目标L1/L2未完成。新P1 BASE private_data/closure_mechanism_20261019/P1/edge_memory_discrimination01；P0全部证据在同名P0目录，原模型/失败/大资产保留。heartbeat ACTIVE30min。
