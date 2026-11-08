# E2 技术报告(阶段 3):整块必要性审计(prune)

> 历史保留标记（2026-09-22）：本报告保留为早期 F1 prune/开发折记录，不能单独代表当前准入结论。
> 相关后续见 [0001](accepted/0001_pruning_horizons_and_module_adoption.md) 与 [0009](accepted/0009_five_fold_refit_acceptance.md)；当前状态见 [current_state](../docs/current_state.md)。

日期: 2026-09-04 · 状态: **PASS** · 证据: `results/e02_prune/{summary.json, model_pruned.npz}`
(F1,generator_10,ZOH;val 43 事件;test 未打开)

## 1. 方法

对全模块联合拟合模型做**整块后向删除**审计:自动删 realized-rank=0 端口,
然后对每个剩余模块实例,在冻结 ridge(1e-8)上删除并**整块重拟合**,
用 source-stratified 2% 非劣门判定是否允许删除。门:对每个 source ×
{1s,5s,10s} 单元,删除导致的 SSE 退化点估计与 95% fault-line bootstrap
上界必须 ≤ 2%(本次探索用 60 次 bootstrap;E6 全量将用 200)。

## 2. 结果

**自动删除**:`resonant_memory f=1.70Hz` —— 联合拟合后 realized rank=0
(零端口),自动移除。这与 AFSD v8 低秩谐振参数不可辨识的旧结论一致:
1.7Hz 通道在该激励集下无支持。

**6 个试删 trial**(每个都整块删除并重拟合):

| 试删实例 | 删除后 val loss | 删除后 raw RMSE | 最差源点退化 | 判定 |
|---|---|---|---|---|
| NODE_HIGH_ORDER τ0.2 | 2.228 | 0.180 | highord 5.9% / dense 16.4% | **保留** |
| EDGE_MEMORY τ0.02 | 1.931 | 0.167 | dense 3.8% | **保留** |
| EDGE_MEMORY τ0.22 | 2.066 | 0.173 | dense 2.0%+ | **保留** |
| EDGE_MEMORY τ1.62 | 1.990 | 0.174 | dense 10.1% | **保留** |
| RESONANT 0.20Hz | 2.163 | 0.176 | highord 22.2% | **保留** |
| RESONANT 0.80Hz | **1.638**(总体改善) | **0.158**(最优) | **highord 39.0%** | **保留** |

所有试删在至少一个 source×horizon 单元超过 2% 非劣门 ⇒ **无一可删**。

## 3. 科学解读(重点)

- **F1 最小有效模块集 = 全部 6 个激活实例**:每个模块在某类源上都是必要的。
  这是模块化 SINDy"选择即删除审计"的直接产物——不靠先验裁模块,而靠
  source-stratified 反事实删除决定。
- **分层门的关键作用**:RESONANT 0.80Hz 是个"完美的反例"——若只看**总体**
  val loss,删除它反而从 1.894 降到 1.638(更"简洁更好");但**分层审计**显示
  highorder 源(稳定长轨迹,20 事件)退化 39%。0.8Hz 正是这批稳定事件的
  区间振荡分量;总体被 dense 短事件稀释掩盖。**没有分层门就会错误地删掉
  对主要稳定场景最重要的谐振通道。** 这是论文方法论的强论证点。
- dense 源只有 1 条 fault line(23 事件),bootstrap 无行内变异(upper=point),
  门退化为点估计——审计注释应注明此局限;E6 用更多折的 fault-line 聚类
  补强。

## 4. 最终 F1 模型

`NODE_HIGH_ORDER + EDGE_MEMORY{0.02,0.22,1.62s} + RESONANT{0.20Hz,0.80Hz}`
- val H100 mean RMSE **0.2446**,pooled ω R² **0.702**,pooled δ R² 0.944,
  worst-10% CVaR 0.569,失败率 0。
- 物理审计 pass:非负、互逆端口、能量供给恒等式按构造成立、storage Hessian
  最小特征值 0.33–0.99(正定)。

## 5. 下一步

进入 E4:在冻结的 F1 模型上跑完整证书组(解析梯度 vs 数值、离散能量平衡
收敛)+ **免积分稳定裕度**(鞍点临界能量 − 初态储能)+ 统一固定 H100 rollout
派生稳定性标签 → FSR/FPR。
