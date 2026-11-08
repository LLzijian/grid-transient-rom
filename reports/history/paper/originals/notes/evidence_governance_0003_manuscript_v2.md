# 技术报告 v2:证据治理升级(accepted 0003)——手稿证据等级全面重估与策略分叉

> 历史快照标记（2026-09-22）：本文档保留为治理决策记录，不是当前待办或投稿就绪判断。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；已完成改写以 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 为准，结论边界见 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-05 · 状态: **关键节点评估(v2)** —— 在 v1(`manuscript_screen_rewrite_plan_v1.md`,
"证书→筛查")之上,0003 使**证据等级问题比措辞更深一层**。只读观测,未改/未提交任何在途文件。

## 0. TL;DR

- `8db48be`(0002,证书→筛查)之后,并发主循环又落库级推进 `0003`
  (acceptance 记录在途:`reports/accepted/0003_holdout_lifecycle_and_modal_reporting.md`):
  **整个 315 事件规范语料被判为"已消费/污染"**,不再是"一次性确认测试"。
- `configs/test_release_v1.json`:`status=blocked_requires_fresh_holdout_and_refit_models`;
  被污染 build id `canonical_rebuild_20260722T022739`;**解锁需一个"全新数据集 build"**(不在污染
  清单)+ 冻结分析 + 全量模块门 refit + 一次性读取账本。
- 后果:**当前仓库里没有任何数字可以当 "confirmatory OOS" 引用**。手稿全部 test 级数字
  (315 pooled 精度/分难度、基线-on-test、0/51+coverage、角度扫描)都是**探索性**;
  modal ~7% 频响 headline 用了 best-pair+验证事件+门禁前模型 → **非确认性**,修正协议未跑出数。
- v1 改写计划(证书→筛查)**必要但不充分**。手稿要 submission-ready,须做**两级 claim 架构**
  (探索性结果 + 声明的 confirmatory 协议/未来 fresh holdout),这需要你定一个策略方向(§5 分叉)。

## 1. 上游新权威(已在途,以落库为准)

1. **0003 三主张**:(a) confirmatory holdout 事件 payload 至多读一次、须先预注册全部端点;
   (b) 已被探索性脚本读过的 build **永不可**重标为 fresh confirmatory;(c) modal 必须用最终剪枝
   模型 + 固定完整 Hungarian 分配的**全对**,不得逐事件取最低误差对;selection validation 不叫
   confirmatory OOS。
2. **实现**:`SinglePassHoldout`(授权前拒污染 build/越种群/重复读,读前记账);删除独立重开 test
   的 `run_e6b_test.py/run_e6c.py/run_e8.py`;`run_test_release.py` 为唯一未来 confirmatory reader
   (真实配置 exit 2,不建 store、不读 payload);E5/E5b/E5c → `run_modal_validation.py`(最终模型、
   全对、标"post-selection validation diagnostic");删除全部 per-event best-match 路径。
3. **0003 回归(证据)**:单事件对误差 0.01 与 0.49 → best-pair headline 0.105、all-pair event-macro
   0.225。**best-pair 可把误差压低 ~2×**(挑中最优对是 cherry-picking)。
4. **STATUS.md 新增段**:315 test build "reopened by E6b/E6c/E8 → consumed exploratory evidence,
   not a one-time confirmatory release";legacy modal headline 非确认(selection val events +
   pre-final full models + per-event lowest-error match);修正 modal 协议只能算 validation
   diagnostic,直到 fresh holdout 可用。
5. **test_release_v1 契约**:angle limit π/2(=0.5·π);`candidate_energy_role: heuristic_secondary_
   screen_not_certificate`;modal 主指标 = event-median of all-pair Hungarian macro,`best_pair_
   selection: false`;`post_release_angle_or_threshold_sweeps: prohibited` on confirmatory holdout。

## 2. 现有证据只剩两级

| 等级 | 可用数字 | 状态 |
|---|---|---|
| **Confirmatory**(预注册+一次性读取+fresh build) | **无** | fresh build 缺失 → blocked |
| **Development / 选择期 OOS(validation folds)** | F1–F3 模块抬升、val 分难度轨迹精度、模块门 | 用于模型**选择** → 非 confirmatory,但可作方法学证据 |
| **Exploratory(被消费的 315 test / legacy modal)** | pooled/分难度 315 精度、0/51、coverage 0.231/0.299、角度扫描、~7% 频响 | 描述性;永不可重标 confirmatory |

**推论**:以 "315 test = OOS 评估" 为骨架的手稿(abstract/贡献/Results)目前**没有确认性 OOS 主张**。

## 3. 手稿 claim 伤亡地图(v1 已列 证书→筛查 之外,新增)

- **C-a 假声明**:Results 契约 "…the pooled full-difficulty numbers are from the **single read** of
  the frozen test folds"(行 ~503)——E6b/E6c/E8 各自重开过 test → "single read" 不成立,**必须删/改**。
- **C-b 315 pooled/分难度精度 + 基线-on-test** → 探索性;表 1 分难度与 "ALL stable 264" pooled
  行不能当 confirmatory OOS headline。
- **C-c 安全**:0/51、coverage 0.231/0.299、41/10、角度扫描鲁棒性(E8)→ 探索性;且 confirmatory
  holdout 上**角度/阈值扫描被禁**(v1 里 0.23→0.30 表不能升级为确认)。
- **C-d modal/freq**:abstract(73–74)与贡献(176–177)"~7% across 212 out-of-sample **validation**
  events";Results(721–729)"**per-event best** frequency error … pooled median 7.2%(fold 4.0–10.7%)"
  与阻尼 550 对 MAE 0.040 —— 正是 0003 判非确认的 legacy 协议(best-pair+val+门禁前模型)。
  0003 回归证明 best-pair 可约 ~2× → **7.2% headline 不可信为诚实的全对值**;修正 all-pair 值
  **尚未跑出/未提交**。
- **C-e 卖点句**:contrib 157 "…extrapolate oscillation frequency";abstract 156 "…predict
  electromechanical frequency" 等,凡绑定 best-pair/315 者同降级。
- (对照)可保留为**方法学/选择期**证据:模块准入删除审计(2% 非劣性、跨折稳定)、能量结构构造+
  本地审计、F1–F3 val 分难度精度与模块增益、因果契约实现、负对照方向结论、"探索性 test 上的
  描述性筛查 0/51(9 组)"。

## 4. 现在能写 vs 不能写(数字口径)

| 主张 | 现在可写(诚实口径) | 不可写/需前置 |
|---|---|---|
| 模块准入、能量结构审计、负对照 | ✅ 方法学 + val 证据 | — |
| 探索性轨迹精度(315) | ✅ 标 exploratory / "consumed test, descriptive" | ❌ 不可称 confirmatory OOS |
| 筛查 0/51(9 组,CP 0.28) | ✅ 描述性(回顾性,test-reuse 标注) | ❌ 不可称保证 |
| coverage/角度扫描 | ✅ 仅 exploratory(注明 sweep 在 confirmatory 上被禁) | — |
| 频响 ~7%/阻尼 0.04 | ⚠️ 仅 legacy best-pair 值(须标非确认) | 需 `run_modal_validation` 全对值重跑后才可作 validation diagnostic 引 |
| 任何 confirmatory OOS 精度/安全 | ❌ | 需新 dataset build + 0001/0002 refit + `run_test_release` |

## 5. 策略分叉(需你定)

- **出路 1(推荐):把论文定位为"方法学 + 探索性证据 + 预注册确认协议"**。重写后骨架:贡献 = 模块化
  符号识别流程 + 能量相容结构/本地审计 + 免积分候选能量筛查(描述性)+ 因果契约与负对照;把 315
  结果标探索性;把 fresh-holdout confirmatory(单次读、全对 modal、禁 sweep)作为明确"后续工作/将
  在全新 build 上执行"。优点:诚实、现在可完成、不需新数据;代价:放弃"确认性 OOS headline"卖点,
  大改 abstract/贡献/Results 骨架。
- **出路 2:先造一个新 dataset build**(全新 39 母线事件,不含被污染 id)→ 冻结分析 → 0001/0002
  gate 下 refit 最终模型 → `run_test_release` 单次读出确认数。工程量大、依赖新数据可得性;成功后
  论文有真 confirmatory OOS。
- **混合**:正文按出路 1 重定位;同时先把 0003 修正 modal(全对 event-macro)在现有最终模型上以
  "validation diagnostic" 跑出诚实数替换 legacy 7.2%(即便仍是 val 级,也去掉了 best-pair 偏差)。

## 6. 手稿改写计划升级(v2 增量)

在 v1(`manuscript_screen_rewrite_plan_v1.md`,E1–E15/C1–C7,仍全部适用并先做)之后增补:

- **E16/C8**:删 Results 契约 "single read"(C-a),改为 "…pooled full-difficulty numbers are an
  exploratory screen of a canonical test build that earlier analyses also read; a preregistered
  fresh holdout (Sec. boundaries) is required for confirmatory inference."
- **E17/C9**:把 315 精度/安全/角度扫描处标 "exploratory"("consumed test build; descriptive")。
- **E18/C10**:modal 节与 abstract/contrib ~7% → 标 legacy per-event-best、non-confirmatory;引
  corrected all-pair protocol;若跑出全对值则替换 headline。
- **E19/C11**:新增/扩写 "Confirmatory protocol and open evidence" 小节/边界:污染 build id、
  test_release_v1 预注册端点、单次读账本、fresh build 前置、sweep 禁令。
- v1 的 D1(标题)/D2(摘要长度)决策点 + 本报告 §5 分叉合并为一个决策清单。

## 7. 决策点(建议默认)

1. **走哪条出路(§5)**:默认出路 1(重定位为方法学+探索性+预注册协议),main.tex 可在干净树上完成;
   出路 2 需先确认"新 dataset build"可得。
2. 是否先把修正 modal 全对值(validation diagnostic)跑出来替换 legacy 7.2% 再定稿(属新实验,
   需最终模型带 `module_adoption_gate` 元数据 → 前置 refit)。
3. v1 D1/D2(标题、"screen,not certificate"摘要句)沿用默认。

## 8. 边界与溯源

- 本报告只读;工作树当前被 0003 在途批(python 改动+文档)占据,未提交;门跑测待其落库。
- 行号以当前 `paper/main.tex`/`main_CN.md` 工作树为准;权威口径在 `reports/accepted/
  0003_holdout_lifecycle_and_modal_reporting.md`(在途)、`configs/test_release_v1.json`、
  `paper/STATUS.md`、`results/README.md`。
