# 论文数字审计报告(投稿前核验节点 v1)

> 历史快照标记（2026-09-22）：本文档仅记录早期数字/主张核验，不代表当前投稿就绪或证据等级。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；后续边界见 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 与 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-04 · 状态: **完成** —— 对 `paper/main.tex`(v1.1)的**数字/主张↔证据
逐项审计**,全部对回不可变 `results/` 与合成主表;发现并修正 2 处(1 列表头误标 + 1 处
H800 评估子集未披露);英文稿**已重编译通过(8 页)**,中文简稿同步。此报告即投稿前
"抽查任意数字"的记录凭证。
修订(v1.1b):§5 中 §V-F "frequency tightens on harder events" 经 E5b 记录**证据定案
为方向错误并已改正**(见 §5);英文稿重编译 exit 0、8 页。至此审计发现的全部问题已闭环,
稿件无已知未决数字/表述问题。

## 0. 审计方法
不改模型、不新读 test 轨迹(除审计已存产物),只把稿件**已打印的数字**程序化对回
`results/e0*/summary.json` 等不可变 JSON + 主表 `reports/synthesis_paper_main.md`。
每条给 PASS/FLAG + 出处。

## 1. 表 tab:select(跨折选择) — PASS
| 稿件 | 源 | 核对 |
|---|---|---|
| F1 val n=43, med 0.50 | e09 F1.val n=43, stable_median_peak=0.50 | ✓ |
| F2 37 / 0.37 · F3 42 / 0.88 · F4 43 / 1.32 · F5 47 / 2.24 | e09 各折 val | ✓ 逐格一致 |
| F1 基座 ωR² −0.02 → 模块 0.702 | e07 M0 pooled_omega_r2 −0.024;modular 0.702 | ✓ |
| F2 0.595→0.748、F3 0.442→0.802、F4 0.459→−0.031、F5 −3.99→−2.77 | e06 各折 summary(先前合成报告抽取) | ✓ |

## 2. 表 tab:safety(安全审计)与图3 — PASS
- π/2 列:certified 292、stable-safe 61、rejected/refused 203、missed **0/51**、
  unstable-called 41、refused 10、coverage 0.231
  → 对 e08 audit `0.500pi` 与 e06 margin_oos(逐折加总 n_stable=264、unstable=51、
  certified 292、safe 61、called 41、refused 10、missed 全 0) **逐格 ✓**;
  fig3 图例(61/190/41/0/13/10)程序化核验一致 ✓。
- 0.9π 列:79 safe、185 rejected、missed 0/51、41/10 不变、coverage 0.299
  → 对 e08 audit `0.900pi` **逐格 ✓**。
- 证书(center 0、grad ≤3e-10、Hessian λmin 0.1265、supply ≤5e-9 全折 PASS)与合成主表一致 ✓。
- §V-D 散文 "18 of 292 certified had smaller margin at wider angle" → e08
  monotonicity: n_compared 292、n_margin_decreased 18、n_became_stable 18 ✓。

## 3. 表 tab:baselines(基线) — 1 处修正
- 各模型 in-domain RMSE 内洽(表注 δ/ω 拆分):
  M0 0.071=√((0.018²+0.099²)/2) ✓、modular 0.045=√((0.012²+0.063²)/2) ✓、
  PINN 0.124 ✓、LSTM 0.346 ✓;mild n=17+mod n=3=20、severe 14 → 34 与 e09 F1 val
  stable 34 一致 ✓;SINDy 发散 36/43 = e07 sindy n_rollout_failures 36 ✓。
- PINN 点态 best_validation_loss 0.122 ✓;"1.6×"/"2.7–7.7×" = 0.071/0.045、
  0.124~0.346/0.045 ✓。
- **FLAG→FIX**:表头 "Pooled speed $R^2$" 与数值不符——该列实为**合并 Δ+ω 的
  pooled R²**(e07 `pooled_r2`:M0 0.313、modular 0.797、PINN 0.278、LSTM −0.90、
  SINDy −1.35e4),而非 speed-only ωR²(M0 应为 −0.02)。已把表头改为
  "Pooled closed-loop $R^2$ (all 43)",数值不变;中文简稿同改。
  (ωR² 本就由表 tab:select 单独报告,不冲突。)

## 4. 表 tab:traj(一次性 OOS test 分难度) — PASS(含出处提示)
- 计数自洽:mild 69 + moderate 40 + severe 155 = 264 = 5 折 test 稳定事件和
  (margin_oos 逐折 n_stable 52+53+50+53+56);严重主导与 e09 test
  stable_median_peak 2.17–3.24 一致;δ/ω RMSE(0.013/0.069、0.046/0.190、
  0.378/1.231、all 0.290/0.947)为 E6b 一次性 test rollout 池化,规范拷贝于合成主表;
  各折聚合源在 `results/e06/F*/test/summary.json`(只含池化,不含难度分层)。
  → 数值与主表一致 ✓;注明:难度分层的逐事件池化未另存单文件,若需复算请以
  E6b 脚本 + 冻结模型重放(不改冻结,重放 test 属审计例外)。

## 5. §V-F 模态外推(频率/阻尼/MAC) — PASS
- 7.2%(pooled per-event best, n=186/212)、单折 4.0–10.7%、<10% 占比 65%
  → e05b pooled median 0.0722、n_matched 186/212、fraction_below_10pct 0.645、
  折 medians F1 4.0…F5 10.7 ✓。
- 阻尼:550 对上 model 0.100 vs emp 0.094、偏差 +0.001、|Δζ| 0.040
  → e05c pooled n=550、0.1000/0.0943、mean_model−emp 0.0011、abs_dzeta_median
  0.0402 ✓。
- MAC 0.39–0.51 → e05b 折 median 0.391–0.510 ✓。
- **FLAG→FIX(v1.1 修订,证据定案)**:§V-F 一句 "Frequency extrapolation tightens
  naturally on harder events" 方向错误,且与后文自相矛盾("若更难更准,pooled 应更低
  而非 7.2%")。裁定依据(E5b 主报告 + e05b 数据):单折中位误差**随折难度上升**
  (F1 温和 4.0% → F5 近临界 10.7%),即更难事件频率更不准——与 E6 轨迹精度难度条件化
  同向;E5b 的"精度自然收紧"是中文歧义(实指"该带的精度上界收紧/精度变差")。
  英文稿已改为: "The single-fold frequency error rises with fold difficulty---4.0\%
  on the gentle F1 against 10.7\% on the near-critical-dominated F5---so we quote the
  pooled ~7\% over the favorable gentle-fold 4\% as the honest headline."
  CN 简稿本就用词正确("折间 4.0–10.7% 随难度"),未含错误分句,无需改。

## 6. 表 tab:pm(Pm 门控,H100/H800)— 1 处披露补齐
- H100 拒绝 −2.1% = e03 validation_improvement_vs_ZOH −0.0209 ✓;oracle ~1% =
  e03 zoh 0.24778 vs oracle 0.24510(全 43)=1.08% ✓。
- H800 采纳 +35.3% = e03b 0.3529 ✓;raw rmse 0.0182→0.0118 = fallback
  raw_mean_rmse 0.01825 → pm_ODE 0.01180 ✓;74% = (0.018235−0.011800)/
  (0.018235−0.009648) ✓;τ≈2.0 s(spectral_abscissa −0.5)+ governor ~2.1 s;89 非零
  系数(active_coefficients 89)✓。
- **FLAG→FIX**:e03b 的 H800 门评估子集 = **n=20 稳定长时程事件**(dense 近临界排除,
  "long-horizon rollout diverges by construction"),而原稿未披露 n 与子集边界(读者会
  误以为全 43 val)。已在 §V-G 与表注补明:"H=8 s gate evaluated on the n=20 stable
  events with long traces; near-critical events excluded…"。H100 门为全 43 val 不变。

## 7. 编译与同步
- 修正后 `latexmk -pdf` → exit 0、**8 页**,无 undefined citation/ref(唯一 benign
  balance warning)。
- `main_CN.md` 同步:基线表头改 "pooled 闭环 R²(全 43 val)"、Pm 表下补 n=20 子集
  说明。

## 8. 待作者后续(非自动可办)
1. §V-F "tightens naturally on harder events" 分句口径(E5 单折 vs 跨折)——见 §5 注意。
2. 作者块/资助、bib 卷期页对照出版商记录、图人眼 QA(本模型无法预览图片)。
3. tab:traj 难度分层池化如需"可一键复算"产物,可加一个只读 E6b 重放脚本(审计例外)。
