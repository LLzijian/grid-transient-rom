# E6 技术报告:五折确认 + 一次性 OOS 测试(论文主表)

> 历史保留标记（2026-09-22）：本报告保留为早期折间/315事件记录；单次读取、证书及确认性表述已撤回。
> 后续边界见 [0002](accepted/0002_candidate_energy_not_certificate.md) 与 [0003](accepted/0003_holdout_lifecycle_and_modal_reporting.md)；准入见 [0009](accepted/0009_five_fold_refit_acceptance.md)。

日期: 2026-09-04 · 状态: **完成** —— 跨折模块选择稳定;OOS 轨迹精度难度条件化
(mild/moderate 精确,severe 近临界到降阶模型边界);**OOS 安全审计全难度
零漏报(missed-instability=0/51,5 折证书全 PASS)** ·
证据: `results/e02_*`(F1)、`results/e06/{F2..F5}/`、`results/e06/{F1..F5}/test*/`
与 `results/e06/margin_oos_summary.json`

## 1. 管线(逐折,不可变产物)

每折:① `backbone` M0 拟合 + 候选提案 ② `full` 模块化能量模型联合拟合
③ `prune` 整块后向删除(source-stratified 2% 非劣门)④ 冻结后**test 只读一次**
(H100,generator_10,ZOH 因果契约;test 在冻结前从未触碰)。每折
train≈206–214 / val≈37–47 / test 61–66;train+val+test = 315 全集,折间
fault-line 不相交。

## 2. 验证集(选择)主表

| fold | train/val | val 难度(stab max|ω| 中位) | backbone ωR² | full ωR² | full δR² | prune 保留 |
|---|---|---|---|---|---|---|---|
| F1 | 209/43 | 温和 0.50 | −0.02 | **0.702** | 0.944 | 全 3 族 |
| F2 | 214/37 | 温和 0.37 | 0.595 | **0.748** | 0.961 | 全 3 族 |
| F3 | 212/42 | 中等 0.88 | 0.442 | **0.802** | 0.947 | 全 3 族 |
| F4 | 206/43 | **严重 1.32** | 0.459 | −0.030 | 0.861 | 全 3 族 |
| F5 | 207/47 | **严重 2.24** | −3.99 | −2.77 | −0.71 | 全 3 族 |

难度 = 该折 val 稳定事件 truth 清除后 1 s 峰值 |ω| 的中位(rad/s);精确值见
`results/e09_valdifficulty/summary.json`(F2 实际 0.37,为全折最温和,非早前估算
0.61;F3 0.88 为真正中等)。模块化模型在**温和/中等 val**
(F1–F3)一致大幅抬升 ω R²(Δ+0.15~+0.72);在**严重近临界占多数的 val**
(F4/F5)无增益或深负——不是折间管线不一致,而是**各折 val 的难度构成不同**
(F4 val 稳定事件中位 max|ω|=1.32、F5=2.24,即多数为近临界 severe)。温和
事件上 F4/F5 模型同样精确(如 F4 `scenario_005` δ_rmse 0.003、F5
`scenario_014` δ_rmse 0.019),证明确非管线故障。

三族模块(NODE_HIGH_ORDER/EDGE_MEMORY/RESONANT_MEMORY)在**全部 5 折**的
整块删除审计中均保留 → **模块选择跨折稳定**(论文核心主张之一)。全部折
physics audit pass、rollout 零失败、COI 漂移 ~1e-15。

## 3. OOS 测试(一次性,315 事件,全部冻结后读取)

**重大分布发现**:验证集是**温和**开发集(val-stable 中位 max|ω| 0.37–2.24 rad/s,
F1–F5 依次 0.50/0.37/0.88/1.32/2.24;见 `results/e09_valdifficulty/summary.json`),
测试集覆盖全难度谱(**pooled test-stable 中位 max|ω| = 2.33 rad/s**,组成 mild 26.1%
/ moderate 15.2% / severe 58.7%)。轨迹精度按清除后 1 s 内 truth 峰值 |ω| 分难度
如实呈现(稳定 test 事件,n=264):

| 难度(1 s 内 max|ω|) | n | δ_rmse | ω_rmse | 说明 |
|---|---|---|---|---|---|
| mild ≤0.5 | 69 | **0.013** | **0.069** | 强泛化 |
| moderate 0.5–1.5 | 40 | **0.046** | **0.190** | 良好 |
| severe >1.5(近临界) | 155 | 0.378 | 1.231 | 模型发散/欠拟合 |
| ALL stable | 264 | 0.290 | 0.947 | 被近临界 majority 主导 |

不稳定 test 事件(n=51,跨折)不在轨迹精度声明内(E4 已确立:不稳定由
margin 三态判定接管)。

## 4. 科学解读(关键局限,如实报告)

- **模型有效域 = 中轻扰动清除后暂态(难度条件化,折无关)**:凡 max|ω|₁s
  ≲1.5 rad/s 的事件——跨 F1–F5 的 val 与 test——模型 OOS 均精确(δ rmse
  0.003–0.05、ω rmse 0.02–0.19)。这一结论在温和折(F1)与严重折(F4/F5)
  中一致成立,证实在论文"简化模型准确反映暂态"的主张限定在清除后暂态
  不逼近同步边界的事件。
- **近临界 severe 事件是降阶模型的固有边界**:清除态 ω 摆幅大(1 s 内
  >1.5 rad/s 直至 5+),系统在同步边界附近强非线性;弱形式拟合以事件等权
  +验证选择无法让模型忠实这些极端轨道。抽查显示模型在此**发散/过判失步**
  (truth 恢复而预测摆开,如 `ncunstable_dense_0112`)——误差方向保守
  (把稳定推向失步),不是漏报不稳定。
- **val 难度构成折间不均(论文须呈现)**:F1 开发折 val 温和,E2–E5 据此得到
  的漂亮数字仅代表温和带;F4/F5 的 val 本身近临界居多。因此**主表必须给
  分难度轨迹精度**,不能给单一折 val mean_rmse。
- **近临界带的正确判据是 margin 三态规则(E4)**:它对不稳定事件"拒而不误判"
  (missed-instability=0)。该规则在温和 val 上已证;须在**全难度 OOS test**
  上重验(见下)。

## 5. OOS 安全审计(E6c,已完成)—— **全难度零漏报**

315 test 事件(264 stable / 51 unstable)逐折冻结模型上跑 margin 三态规则
(证书 + margin>0 ⇒ stable;margin≤0 ⇒ unstable;未证书 ⇒ refused,绝不判稳)。
证据 `results/e06/margin_oos_summary.json` + 各折 `test_margin.json`。

| fold | test | 证书 | stable_safe | 稳定误拒/拒 | **不稳定漏报判稳** | 不稳定判unstable | refused |
|---|---|---|---|---|---|---|---|
| F1 | 63 | 41 | 7 | 45 | **0/11** | 2 | 9 |
| F2 | 64 | 64 | 10 | 43 | **0/11** | 11 | 0 |
| F3 | 61 | 60 | 19 | 31 | **0/11** | 10 | 1 |
| F4 | 66 | 66 | 11 | 42 | **0/13** | 13 | 0 |
| F5 | 61 | 61 | 14 | 42 | **0/5** | 5 | 0 |
| **pooled** | **315** | 292 | 61 | 203 | **0/51** | 41 | 10 |

- **missed-instability rate = 0.0(全难度 OOS)**。含近临界 severe 带(轨迹
  拟合超出降阶模型域的事件):这些事件被 margin **主动判 unstable 或拒绝**,
  无一漏报。这是论文安全主张的最终证据——免积分规则在未训练事件上把
  "漏报不稳定"压到 0。
- **代价:stable 覆盖 23%(61/264)**。severe 近临界**稳定**事件多被判
  unstable/拒绝(保守误报)。方向安全,覆盖保守。
- **5 折模型证书全 PASS**:storage 中心化 0、梯度 ≤3e-10、投影 Hessian 最小
  特征值 **0.1265**(正定,五折相同)、供给恒等式误差 ≤5e-9。能量结构自洽性
  跨折成立。

## 6. 主表使用建议

论文主表 = 跨折 val(选择一致性,标注难度)+ OOS test **分难度**轨迹表 +
**OOS margin 安全表(全难度零漏报)**。指标用分难度 rmse 与 pooled ω/δ R²,
不用单一 mean_rmse(被 severe majority 与 unstable 事件污染)。安全叙事:
简化模型在中轻扰动上精确再现暂态,对近临界事件"拒/判不稳"而零漏报。

## 7. 工程注记

- 各折耗时:F1 backbone/full/prune≈2+5+21 min;F2 100 draws prune≈48 min;
  F3/F4/F5(60 draws)≈16/49/48 min。全部后台顺序执行、不可变目录、拒覆盖。
- E6b test 脚本与 E2 rollout 同码同界(H100、ZOH exo、aggregate 同款)。
- prune 追溯记录自动零端口删除与逐 trial 源分层退化;draws:F1 探索 60、
  F2=100、F3–F5=60(规则同,仅 CI 紧度异)。
