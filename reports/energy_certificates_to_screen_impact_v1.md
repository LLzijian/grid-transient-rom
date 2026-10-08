# 技术报告:能量"证书"→本地结构审计 + 启发式候选能量筛查(并发主循环在途变更)对手稿声明的影响清单

> 历史快照标记（2026-09-22）：本文档是已完成语义改写前的历史影响清单，不是当前待办或投稿就绪判断。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；改写已由 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 完成，结论边界见 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-05 · 类型: **关键节点观测/冲突报告**(只读;未改动、未提交任何在途文件)

## 1. TL;DR

并发主循环正在做一次**科学诚实性收敛**:把此前代码/手稿里的"能量证书 / certified 三态判稳"
降级为

1. **本地数值结构审计**(`energy_structure.py`,改名自 `certificates.py`+`energy.py`,二者**删除**);
2. **启发式候选能量筛查**(`critical_energy_level`/`stability_margin` 现在**无条件拒绝正式证书**,
   `certified: False`、`margin: None`、`verdict: "refused"` 恒为真,只保留
   `screening_margin`/`screening_predicted_stable` 作回顾性启发式筛查)。

**该收敛在科学上是正确的**:局部 SLSQP/root 搜索得到的"相邻鞍点+角边界候选能量最小值"是真实
逃逸能量的**上界**而非**下界**,用它判"margin>0 ⇒ stable"缺验证过的全局下界支撑,旧 `certified`
语义不成立(§4 论证)。

**对手稿(`paper/main.tex` v1.1b / `main_CN.md`)的影响是实质性的**:摘要、贡献点、§V 标题、
§V-D 三态规则、表 caption、"Why the margin rule is safe in practice" 等处把该筛查当作
**证书 / 定理**来卖,需改述;而**实证数字全部保留**(0/51、轨迹精度、覆盖 0.23/0.30、λmin 0.1265 等)。
精确冲突清单见 §6。

**门控实测(在途工作树 + HEAD 混合态):G1 ✅ / G2 ✅ 17 passed / G6 ✅ status=pass**。见 §3。

## 2. 在途变更范围(工作树 vs HEAD `f21d34c`)

| 文件 | 变更 | 方向 |
|---|---|---|
| `src/kiki/evaluation/certificates.py` | **删除**(158 行) | `evaluate_certificates` → `energy_structure.evaluate_energy_structure` |
| `src/kiki/evaluation/energy.py` | **删除**(90 行) | 并入 `energy_structure.py` |
| `src/kiki/evaluation/energy_structure.py` | **新增** | `EnergyStructureAudit`,含 `claim_boundary`:"local numerical audit only; no hidden-state curvature, global decrease, region-of-attraction, or escape-energy certification" |
| `src/kiki/evaluation/stability.py` | +77 行 | 新增 `one_sided_binomial_upper_bound`(Clopper–Pearson 精确单侧上界)+ `grouped_missed_instability_audit`(按 `source+fault_line` 独立组做二项审计,避免伪重复) |
| `src/kiki/models/prunable_energy_model.py` | +87/-… | `critical_energy_level()` 返回 `certified: False`、`certificate_status: refused_no_verified_global_escape_energy_lower_bound`、`critical_energy_lower_bound: None`;候选改名 `*_candidate`,带 `stability_boundary_membership_verified: False`、`global_optimization_performed: False`;`stability_margin()` 恒 `margin: None / predicted_stable: False / verdict: refused`,另给 `screening_*` |
| `experiments/run_e4.py` / `run_e6c.py` / `run_e8.py` | 重写语义 | 改 import `energy_structure`;逐事件记 `certificate_status`/`screening_*`;`run_e8` 把**既有不可变** `results/e06/{fold}/test_margin.json` 在内存中降级为 `legacy_candidate_result_demoted_no_verified_global_lower_bound` 后再做角度扫描 |
| `tests/test_stability_semantics.py` | **新增** | `test_positive_candidate_margin_is_not_a_stability_certificate`(正 margin 仍 `verdict=refused`)、CP 上界、分组漏报审计 |

无任何残留 import 指向已删模块(已 grep 核验);`src/kiki/__init__.py`、`baselines.py`、
`stability_region.py` 中的 "certificate" 仅为说明文字,且早已声明 `certificate_claim: False`
/"sample-based evidence, not a formal certificate"。

## 3. 门控实测(venv 复用 kron-reduce,未新建环境)

工作树 + HEAD 混合态(即"下一个 coherent 变更跑门"前一刻的状态):

| 门 | 命令 | 实测 |
|---|---|---|
| G1 syntax | `compileall -q src experiments tests` | ✅ exit 0 |
| G2 unit | `pytest -q` | ✅ **17 passed**(此前 14 + 新增 3) |
| G6 json | `verify_artifacts.py --json-only` | ✅ scanned 74、errors 0、status pass(warnings=92 = 语料合法 Infinity,legacy 不判败) |

注:此绿只说明"在途态语法/单测/artifact 自洽",**不构成对该 refactor 的验收**;验收与提交属
主循环按协议进行。此处未跑 E4/E6c/E8 实跑脚本(会写 results,违反"不代替科学循环"的边界)。

## 4. 为什么"证书→筛查"的收敛在科学上是对的

设 `V*` = 后清除平衡井真实的最低逃逸构型能量(全边界下确界)。局部搜索只找到
`found ⊂ all`,故 `H_c_found = min_{found} ≥ V*` —— **搜索到的候选最小值是 V* 的上界**。

- 判稳的充分条件是清除态储能 `H0 < V*`(此时 H 非增 ⇒ H(t)≤H0 永远够不到井沿 ⇒ 有界)。
- 代码只有 `H0 < H_c_found`。因 `H_c_found ≥ V*`,`H0 < H_c_found` **推不出** `H0 < V*`:
  若真实存在更低、被搜索漏掉的鞍点/边界,则 `H0` 可能落在 `[V*, H_c_found)` 而实际失稳。
- 对偶地,判"unstable"需要 `H0` 高于**验证过的逃逸能下界**;found-min 是上界,也不支撑。
- SLSQP(角边界可行点)与 root(鞍点)都是**局部**方法,不证明搜索完备;`global_optimization_performed: False`
  正是承认这一点。

故旧 `certified=True` 语义(局部候选搜索完备 ⇒ 可发证书)不成立;**新代码对一切正式判决
`refused` 是正确的**,并把"正候选 margin"老实标为启发式筛查。**0/51 是回顾性筛查事实**——旧
规则从未把真失稳事件判稳——它不随收敛消失;其诚实的置信表达改由 fault-line 分组的
Clopper–Pearson 单侧上界承担(见 §5c)。手稿若想恢复"证书",必须先实现真正的验证全局逃逸能
下界(§7 路径 B),新代码已把其显式列为未实现。

## 5. 数字可复现性核对

对照 HEAD `certificates.py` 与新 `energy_structure.py`:`hs`(全状态中心差分)→
`block = hs[:2n,:2n]`(仅角度/速度块)→ `P`(`2n→2(n−1)` COI 投影)→ `eigvalsh` 最小特征值。
**两者数值定义逐行一致**;旧注释里"hidden 以 reciprocal 二次储能故 identity-like,加正地板"
只是**构造性断言**,从未把 hidden 数值折入 λmin。因此:

- λmin=0.1265(每折)与 center≈0、grad≤3e-10、supply≤5e-9 **预计重跑不变**;
- 但手稿 §V-D 行 461–463 "projected Hessian restricted to the **COI quotient and hidden
  coordinates**" 一直超出代码实际数值审计的范围;新审计显式改名
  `angle_speed_projected_hessian_min` 并把 hidden 排除在审计外。**措辞无论如何都要改**(hidden
  只能按构造/互易论证陈述,不能写成被数值审计的量)。

## 6. 手稿受影响声明清单(逐条,引原文+行号)

### A 类 —— 语义不再成立,必须改述(把"certified/证书/定理"降为"候选能量筛查/回顾性审计")

1. **摘要(行 61–64)**: "…a storage function with a numerically certified supply identity,
   which enables an *integration-free* three-state verdict from the clearing state:
   **certified with positive margin (stable), certified with non-positive margin
   (unstable), or refused**."
   → 三态结构可留,但 "certified … ⇒ stable/unstable" 必须改为
   "candidate-margin screen: positive ⇒ screen-stable, non-positive ⇒ screen-unstable,
   uncompleted/refused" 之类;不可再用 certified 断言稳定性。
2. **摘要(行 69–72)**: "…the margin rule yields zero missed instabilities on all 315 …
   (0/51) at a stable coverage of 0.23 (0.30 at a wider search angle), **with energy
   certificates passing on every fold**."
   → 0/51、0.23/0.30 保留(实证);"energy certificates passing" 改 "local energy-structure
   audits pass"。
3. **贡献点(行 170–174)**: "…with certified storage identity, and a three-state verdict
   (stable / unstable / refused) from the clearing state that produces zero missed
   instabilities…"
   → "certified storage identity" 可留(结构恒等式+本地审计);"three-state verdict …
   produces zero missed" 需限定为回顾性筛查。
4. **§V 标题(行 420–421)**: "Energy Structure, **Certificates**, and the Integration-Free
   Margin Rule" → "…Local Energy-Structure Audits and an Integration-Free Candidate-Energy
   Screen"(或类似)。
5. **§V-D2 三态规则(行 469–494,eq:critical/eq:margin 及三条 verdict)**: 把 `H_c`
   (min over found adjacent saddles & angle boundaries)当"the critical energy"、
   "never declares stability **without a certificate**"(行 485)、"**certified and M>0** ⇒ stable"
   (行 487)。
   → 需重述:该 min 是**候选**临界能量(`candidate_energy_semantics: heuristic minimum of
   local boundary and unverified saddle candidates`),正候选 margin 只是筛查;无 verified
   global lower bound 时无正式证书。eq:critical 的 "estimated" 措辞本身已弱,但 verdict 逻辑
   强于证据。
6. **Results 契约(行 504–507)**: "safety metrics are the three-state rule's missed-instability
   rate (…) and stable coverage (fraction of truly stable events **certified stable**)"
   → "certified stable" → "screen-stable"。
7. **Safety 节与表(行 653–679、表 caption 682–699、行 793)**: "41 are correctly declared
   unstable and 10 refused"、"The certificates pass…"、caption "…and **Energy Certificates**"、
   caption 内 "Certificates (all five folds): storage centered 0, …"。
   → 41/10 与 0/51 保留为筛查判决计数;caption "Energy Certificates" 改 "Local
   energy-structure audits";每折审计数值本身保留。
8. **"Why the margin rule is safe in practice"(行 788–794)**: "Because $H$ is non-increasing
   …, a cleared storage **below the lowest escape energy of the post-clear well implies
   bounded motion**."
   → 这是把 found-min 当成了验证过的"lowest escape energy"的定理化论证,**必须重写/删除**;
   诚实版:(i) H 非增 + supply identity 数值审计保留;(ii) 有界结论只对验证过的逃逸能**下界**
   成立,此处改为"在审计语料上筛查从未把真失稳事件判稳(0/51),拒绝方向保守,并给出分组置信上界"。
9. **讨论(行 786 附近 "…that yields a certificate at all" / 行 841 "its margin rule gave zero
   missed…")**: 前者需随 §6 措辞同步;后者 0/51 实证可留。
10. **标题(行 35)**: "…with Integration-Free **Stability Margins**…" → 若正文已诚实降级为
    筛查,标题可保留 "Margins" 但建议弱化为 "…with Integration-Free Clearing-State
    Screening" 或明确 "heuristic margins";由你定夺(见 §8 待确认)。

### B 类 —— 数字保留、但语义标签/审计范围措辞需修

11. **§V-D(行 457–467)** 小节标题 "Numerical energy certificates";行 461–463 "…projected
    Hessian restricted to the COI quotient **and hidden coordinates** is positive definite
    (minimum eigenvalue λmin=0.1265 on every fold)";行 465–467 "We do not assert global
    monotone dissipation or a global region of attraction; **the certificate is** the
    structural identity plus these numerical audits."
    → 该段已相当克制,但(a)改名"本地数值结构审计";(b)"and hidden coordinates"删去——λmin
    数值只审角度/速度 COI 块,hidden 仅构造性论证;(c)补一句"亦不声称验证过的全局逃逸能下界
    (无证书)"。数字 0.1265 等预计不变(§5)。

### C 类 —— 不受影响、可原样保留

- 轨迹精度(RMSE/R²、4.0%→10.7% 频率、阻尼 ~0.04)、模块准入(2% 非劣性删除审计)、
  负对照、因果契约、0/51 作为**回顾性零漏报事实**、覆盖 0.23→0.30 扫描鲁棒性、以及
  新引入的 fault-line 分组 Clopper–Pearson 置信上界(可作为 A7 的量化替代)。
- CN 镜像(`main_CN.md` 行 56–70 §3、95–110 主张②、133、168 证据链)与英文稿逐条对应,
  需同批改述,尤以行 60 "清除储能低于最低逃逸能量则轨有界、判稳" 与行 66 "certified &
  margin>0 ⇒ stable"、行 133 "有(证书+margin)" 为最。

## 7. 建议路径

- **路径 A(推荐,低成本)**:等并发主循环把本 refactor 落库并通过注册门、按新语义重跑
  E4/E6c/E8 生成新的不可变 results 之后,对 `main.tex`/`main_CN.md` 做一轮 §6 所列
  "certificate → screen/audit" 诚实改述;实证数字全保留。此改述是下一个"单一 coherent 变更"
  的天然候选,应 register-claim → 干净 checkpooint → 跑门 → accept/revert。
- **路径 B(长期)**:若要拿回"证书"字样,须实现**验证过的全局逃逸能下界**(完整 UEP/边界枚举
  + 完备性证明,或分支定界/SOS),而非局部搜索;新代码已显式声明未实现。除非你指令,不建议本
  轮触碰。

## 8. 边界与后续处置

- 本报告为**只读观测记录**,未修改/未提交任何在途源文件;G2/G6 绿仅指在途混合态,不构成对
  refactor 的验收。
- 主循环随后以真实事件、17 项测试、结构审计及分组置信界独立验收,并将 refactor 提交为
  `8db48be`;正式口径确定为“本地能量结构审计 + 启发式候选能量筛查”。
- §6 的手稿冲突清单保留为后续独立论文改述循环的输入,不在本观察报告中直接修改正文。

## 9. 溯源

- 关键源行号:`src/kiki/models/prunable_energy_model.py`(critical_energy_level/stability_margin
  返回体)、`src/kiki/evaluation/energy_structure.py`(EnergyStructureAudit/claim_boundary)、
  `src/kiki/evaluation/stability.py`(CP 上界/分组漏报)、`tests/test_stability_semantics.py`。
- 手稿行号以 `paper/main.tex`(v1.1b)与 `paper/main_CN.md` 当前工作树为准;gate 实测命令同
  §3,venv = `E:/03-nn_Transfer_Func/kron-reduce/.venv`。
