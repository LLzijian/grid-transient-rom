# E8 技术报告:margin 安全-覆盖效率(angle_limit 敏感性,全难度 OOS)

> 历史保留标记（2026-09-22）：本报告仅是启发式候选能量筛查的敏感性诊断，不是证书或新判据。
> 证据生命周期以 [0002](accepted/0002_candidate_energy_not_certificate.md) 与 [0003](accepted/0003_holdout_lifecycle_and_modal_reporting.md) 为准；当前状态见 [current_state](../docs/current_state.md)。

日期: 2026-09-04 · 状态: **完成** —— 免积分三态规则对临界能量搜索角
angle_limit 的敏感性审计(315 test 事件全难度):π/2→0.9π 时 stable coverage
**0.231→0.299**(+18 事件转 safe),**missed-instability 保持 0/51**,41 个
不稳定判 unstable 与 10 个 refused **逐事件不变**(无一不稳定翻转)。
margin **非**严格随角度单调(292 获证书事件中 18 个下降),故这是**经验审计结论
而非定理** ·
证据: `results/e08/summary.json`(0.5π 基线复用 `results/e06/{fold}/test_margin.json`,
0.9π 本实验计算) · 仅审计规则,冻结模型未动、test label 仅用于审计

## 1. 目的与定位

E6c 给出全难度 OOS 零漏报(0/51)但 stable coverage 仅 0.231。E4 建议的改进
方向之一即放宽 angle_limit 以获得更多证书。本实验回答两个问题:
(a) 零漏报性质对 angle_limit 是否**鲁棒**(而非只在 π/2 刀口上成立)?
(b) 放宽角度能**买多少覆盖**,代价是什么?

**诚实定位**:这是对审计规则**敏感性/稳健性**的诊断,不是"在 test label 上挑选
操作点"的模型选择——0.9π 是探索性稳健性点,不是用覆盖率最大化反选出来的。
任何"论文应选 angle=X"的声称都要么有物理理由、要么在独立数据上验证;本实验只
证明 π/2 附近的性质不脆。

## 2. 方法与数据

- 五折冻结 **prune** 模型(`e02_prune` F1、`e06/{F2..F5}/prune`),315 test 事件
  全难度(264 stable / 51 unstable)。
- 三态规则不变:证书 & margin>0 ⇒ stable;证书 & margin≤0 ⇒ unstable;未证书 ⇒
  refused(绝不判稳)。margin = critical_energy − storage_at_clearing,免积分。
- 0.5π 基线 = E6c 已算行(E6c 用 π/2);0.9π 由本实验逐事件 `stability_margin`
  计算。critical_energy = min(45 边×± angle-boundary SLSQP, 相邻鞍点 root)。
- margin 单调性以同一事件两角度对比记录(`n_compared` 限于获证书=可算 margin 者)。

## 3. 主表(pooled 315)

| 量 | 0.5π(E6c) | 0.9π(E8) |
|---|---|---|
| 获证书 | 292 | 292 |
| **稳定判稳(stable_safe)** | 61 | **79** |
| 稳定拒/拒绝 | 203 | 185 |
| **漏报不稳定(判稳)** | **0** | **0** |
| 不稳定判 unstable | 41 | 41 |
| 不稳定 refused | 10 | 10 |
| **stable coverage** | **0.231** | **0.299**(+6.8pt,+18 事件) |
| **missed-instability rate** | **0.0** | **0.0** |

**逐事件不变性**:41 个 unstable-call 与 10 个 refused 在 0.9π **逐一保持原判**——
不只是比率,是集合级不变 → 零漏报在放宽角度下不是靠"以别的不稳定换来的",
而是真正无一不稳定事件被 margin 抬到稳定。

## 4. 单调性诚实记录(非单调!)

| 量 | 值 |
|---|---|
| n_compared(获证书事件) | 292 |
| margin 下降的事件数 | **18**(margin 非严格随角度单调) |
| 转为 predicted_stable 的事件数 | 18(= coverage 增量) |

**解读**:critical_energy = min(boundary, saddle) 中,边界能量随角度增而增的直观
是"单方向"的;但更大角度会引入**更远处、能量更低**的新边界/鞍候选,min 可能
下降(18 事件)。这证明**"放宽角度只增不漏"不是定理**——安全必须逐事件在数据上
重验(本实验做了),不能靠理论假设跳过。而实测 0.9π 下净效果仍是 +18 safe、
零翻转,稳健性成立。

## 5. 科学结论

- **零漏报不是 knife-edge**:π/2→0.9π 的 ±perturbation 下,免积分规则对 315 个
  全难度 OOS 事件仍零漏报,且不稳定判 unstable/refused 集合逐事件不变。这是
  安全主张的**稳健性证据**,比单点结果强。
- **覆盖有真实 headroom**:0.9π 把 safe 判稳从 61→79(23%→30%)。方向正确但
  有限——severe 近临界稳定事件仍多被判 unstable/拒绝(保守)。大幅提升需
  **混合判据**(margin 拒绝的近临界事件转有界轨迹复核)或改进临界能量搜索
  (多鞍点/max),均超本实验范围且后者或破坏免积分性。
- **操作点建议(论文)**:维持 **π/2**(最近相邻势阱、物理上可先验辩护)为主,
  把 E8 作为脚注级稳健性报告:同数据上更大角度(至 0.9π)保持零漏报并提高
  覆盖,支持"拒而不误判、覆盖保守"的叙事。**不应**声称"0.9π 是更好的操作点"
  (它是由本 test label 观察到的;选择需独立原则/数据)。

## 6. 诚实边界

- 这是审计规则的**敏感性研究**,非新判据、非新模型;不改变 E6c 的任何主表数字
  (0.5π 行与 E6c 逐位一致:n=315, safe 61, coverage 0.2311)。
- 覆盖提升幅度取决于 angle 与候选搜索;0.9π 覆盖仍 30%,近临界带判稳是未来
  工作(混合判据需重新审视"免积分"卖点与代价)。
- angle_limit 是搜索参数而非物理量;其取值与影响必须与结果同报。

## 7. 工程注记

- `experiments/run_e8.py`:CLI `--angles`(π 倍数)、`--folds`;0.5π 基线自动从
  E6c `test_margin.json` 合并,只算新角度;逐事件 `stability_margin`,写
  `results/e08/summary.json`(不可变、拒覆盖)。
- 成本:0.9π 全 315 ≈ **2.3 h CPU**(≈26 s/事件,远重于 π/2 的 2–9 s/事件;
  大角度边界 SLSQP 远离平衡、非线性强)。分折顺序打印;冒烟先跑 F1 两角度
  校准后收敛为"单角度 0.9π + 基线合并"以控时。
- margin 单调性检查存于 `summary.json["monotonicity_check_margin_in_angle"]`。
