# 新旧S固定开发结果的有限独立只读诊断

复核者classification只读旧artifacts/T4_0001/evaluation_P_S.json、新evaluation_S.json和旧comparison路径元数据，未读模型/缓存/HDF/FOM/TEST，未复算科学指标、不改门槛或选择新模型。一次实际exec_command成本2.2617246秒，外部另计。

新旧均原117/覆盖107/时延10，失稳原22/覆盖12，ID/group/label一致。107唯一数值门全部通过、完整、全状态有限；107行integration_failures的fine/coarse均null，实际失败0。100ms、8s决策、.005/.0025s、原RMS容差.001与hit一致性.02s、TRAIN尺度未改变，源身份未变、TEST未读。

| 指标 | 旧S | 新S |
|---|---:|---:|
| 失稳检出 |9/12|10/12|
| 漏检/错误window_settled |3/3|2/2|
| 假LOS |0|1|
| 数值未决/积分失败 |0/0|0/0|
| severe relative mean/pooled/tail |1.162810/2.403925/3.264001|.977228/1.923580/2.597751|
| 5s relative mean/pooled/tail |.248792/.392178/.820557|.261078/.586798/1.125299|
| 8s relative mean/pooled/tail |.205762/.362450/.718653|.226515/.581020/1.060944|
| severe delta/common/p pooled |.723670/.941791/.482274|.562788/.967025/.378629|
| 5s delta/common/p pooled |.193197/.336211/.150894|.349155/.239208/.147182|
| 8s delta/common/p pooled |.198008/.272040/.120746|.362862/.215542/.124724|

delta单位rad、common rad/s、p pu；tail为最差10%事件RMSE均值。5/8s仅95个真实长区间可评分，另外12个严重短前缀单列，不算长区间合格。

仍漏检且window_settled：pilot_0155(group10,true .81s)、expansion_0155(group34,true .70s)。新检出expansion_0189：true .45s/pred .84s，仍晚.39s。新增假LOS为稳定标签pilot_0237，pred1.48s，数值门通过。pilot_0232仍窗口未定。窗口平息不是稳定证书。

全部12个严重相对RMSE改善，但物理量并非全改善：pilot_0366 relative4.395700→3.487769、delta1.419084→1.100919、omega5.475888→4.373188、p.859354→.667612，而common1.718929→1.820327变差；pilot_0239 relative2.132302→1.707733、delta.273676→.232084、omega2.970356→2.373493、common.453224→.428366、p.298458→.272421。残留漏检pilot_0155 relative1.250034→.960186、expansion_0155 1.523294→1.299053，其改善不能抵消错误判稳。pilot_0063/0152/0157与expansion_0157的p上升。

触发误差旧→新(s，记录时间下pred−true)：pilot_0058 +.12→+.10；pilot_0063 +.03→+.04；pilot_0152 +.20→+.19；pilot_0155漏检不变；pilot_0157 +.12→+.11；pilot_0239 −.02→+.02；pilot_0366 −.36→−.30；expansion_0155漏检不变；expansion_0157 +.08→+.07；expansion_0174 0→0；expansion_0186 +.22→+.19；expansion_0189漏检→+.39。

结论：严重前缀相对改善、少一漏检与common改善具有互补研究价值；新增假LOS及长时域relative/delta/tail恶化不能被均值或其他指标抵消。没有整体高精度、安全性或部署资格声明。
