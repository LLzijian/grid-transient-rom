# 新旧N固定开发结果的有限独立只读诊断

复核者classification仅读取新evaluation_N.json、旧evaluation_P_N_traced.json和旧full_comparison/comparison.json；未读模型、缓存、HDF、FOM/TEST，不重跑指标，不选择新模型。

新旧ID/group/label与原117/覆盖107、失稳原22/覆盖12/时延10相同。107个分类数值门唯一且全部通过、完整、全状态有限；107行integration_failures实际非空失败为0。12严重前缀均可评分且数值合格，步长/门槛/尺度未改。

两者检出10/12，漏检及错误window_settled均为pilot_0155、expansion_0155，真值记录时间.81/.70s，预测无触发。假LOS与数值未决0；窗口平息不是稳定性证书。

| 相对指标 | 旧N400 | 新N800追加阶段 |
|---|---:|---:|
| severe prefix mean |1.039363|1.077374|
| severe prefix pooled |1.806816|1.891651|
| severe prefix worst10% mean |2.777433|2.795989|
| 5s mean/pooled/tail |.373451/.427484/.785097|.369172/.422396/.776518|
| 8s mean/pooled/tail |.312985/.357615/.655772|.309145/.353267/.649223|
| 5s delta pooled rad |.133247|.129752|
| 5s common pooled rad/s |.803785|.753926|
| 5s p pooled pu |.275028|.256550|
| 8s delta pooled rad |.110314|.107446|
| 8s common pooled rad/s |.639347|.604870|
| 8s p pooled pu |.221965|.206947|

5/8s仅95/107个真实区间可评分，12个严重事件缺少足够真值，不以此汇总替代严重前缀。

主要严重事件相对RMSE旧→新：pilot_0155 1.2091→1.4207（delta .3093→.3484rad、relative omega1.5874→1.8778rad/s、common1.1150→1.1114rad/s、p.4260→.4377pu）；pilot_0366 2.8701→3.0093（delta1.0564→1.0567、relative omega3.4122→3.6453、common2.3375→2.3234、p.7750→.7559）；expansion_0155 1.4864→1.5968（delta.3523→.3807、relative omega1.9745→2.1195、common.7286→.7180、p.4268→.4382）。pilot_0239改善2.6847→2.5827（delta.3355→.3154、relative omega3.7436→3.6042、common.6667→.5232、p.3405→.3308）。

expansion_0189真触发.45s，预测.75→.93s，误差+.30→+.48s，相对前缀1.0299→1.1197；expansion_0186误差+.12→+.14s，pilot_0058 +.07→+.09s；pilot_0366 −.30→−.29s。时间均为原记录时间，不偷偷改成初始化后时标。

结论范围：长时域汇总略改善，严重前缀退化、两漏检未解、部分触发延迟增加。仅冻结JSON复核，数值门/缓存并未由第二人重算，不能以均值抵消风险。三条实际工具成本2.0080784+2.4175817+2.3564958=6.7821559秒，另计，不重复受管费用。
