# 技术报告:手稿 "证书→候选能量筛查" 改写方案(apply-ready,逐条原文→改文)

> 历史快照标记（2026-09-22）：本计划记录的改写已完成，不是当前待办或投稿就绪判断。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；执行结果以 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 为准，结论边界见 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-05 · 状态: **改写方案(待应用)** —— 代码与证据侧已完成(accepted 0002),手稿侧是
声明为 not submission-ready 的待办;本报告给出可直接套用的逐条改写。

## 0. 上游现状(已落库的权威口径,改写的唯一依据)

- `8db48be fix: demote unverified energy certificate to screening` + `61ac5be`(验收
  `reports/accepted/0002_candidate_energy_not_certificate.md`)。
- `paper/STATUS.md`:main.tex/main_CN.md "predate the critical energy audit ... not
  submission-ready";历史 `certified`/`margin`/`zero missed` 必须解读为**回顾性启发式筛查**,
  非定理/安全保证;须反映 **0/51 描述性 + 9 独立组 + 单侧 95% 精确上界 0.283129** 与 **test-reuse
  限制**。
- 不可变 results **未重跑**(immutability 保持);`run_e8` 在内存把旧 `certified` 行降级为
  `legacy_candidate_result_demoted...`。故改写在**纯散文语义层**,不改任何数字(0/51、61/79、
  0.231/0.299、λmin 0.1265 等均保留)。
- 权威实测数值(0002,F1 两例):中心化 0;梯度 2.52e-10;角度/速度 COI 投影 Hessian 最小
  0.126528;局部供给恒等式误差 1.83e-10;决定性回归例 `ncunstable_dense_0006`:候选 0.128886、
  筛查 margin 0.126537、certified=false。分组口径:**0/51 = 9 个独立 source+fault_line 组**,单侧
  Clopper–Pearson 95% 上界 = **0.283129**。

**改写总原则**:凡把筛查写成"证书/判稳定理/安全保证"处 → 降为"候选能量筛查 / 本地结构审计 /
描述性实证";**数字与事件划分全部保留**;补"非验证全局下界、非零风险、test-reuse 限制"。

---

## 1. 英文稿 `paper/main.tex`(行号以当前工作树为准)

### E1 摘要 · 句块(行 ~60–72)
> 原文:…the resulting model retains a storage function with a numerically certified supply
> identity, which enables an \emph{integration-free} three-state verdict from the clearing state:
> **certified with positive margin (stable), certified with non-positive margin (unstable), or
> refused**. …the margin rule yields zero missed instabilities on all 315 full-difficulty events
> (0/51) at a stable coverage of 0.23 (0.30 at a wider search angle), **with energy certificates
> passing on every fold**.

改文(需在摘要字数内压缩):
> …the resulting model retains a storage function whose local energy structure (centering,
> stationarity, angle/speed curvature, supply identity) we audit numerically, and whose lowest
> \emph{found} boundary and saddle energies yield an integration-free candidate-energy margin from
> the clearing state. A local candidate search does not establish a verified global lower bound on
> escape energy, so the margin is a heuristic screen, not a certificate: …the screen never declared
> a truly unstable event stable on the 315 full-difficulty events (0/51, spanning 9 independent
> groups) at 23\% stable coverage (30\% at a wider search angle), and the local structure audits
> pass on every fold.

### E2 贡献点 · bullet 3(行 ~170–174)
> 原文:…with certified storage identity, and a three-state verdict (stable / unstable / refused)
> from the clearing state that produces zero missed instabilities on the full-difficulty out-of-sample set.

改文:
> …with a numerically audited storage identity, and an integration-free candidate-energy screen
> whose clearing-state verdicts (stable / unstable / refused) never declared a truly unstable event
> stable on the full-difficulty out-of-sample set (0/51).

### E3 引言 · 动机句(行 ~121–122)
> 原文:…carry no structural guarantee that could anchor a stability certificate.
改文(避免预告我们也不提供的证书):…carry no structural guarantee that could anchor a stability
> judgment.

### E4 §方法预览(行 ~263–265)
> 原文:The retained model carries an energy certificate and an integration-free stability margin,
> which together support the deployment verdicts …
改文:The retained model carries an audited energy structure and an integration-free
> candidate-energy margin, which together support the screening verdicts …

### E5 §V 标题(行 ~420–421)
> "Energy Structure, **Certificates**, and the Integration-Free Margin Rule"
改文:"Energy Structure **Audits** and the Integration-Free **Candidate-Energy Screen**"

### E6 §V-D1 小节(行 457–467)
标题 "Numerical energy certificates" → "**Local energy-structure audits**"。
正文(458–467)改文:
> For each frozen model we audit, relative to the post-clear equilibrium, that the storage locally
> has the claimed shape: it is centered ($|H(\delta^{\star})|\approx0$), its gradient vanishes
> ($\|\nabla H\|\leq3\times10^{-10}$), the center-of-inertia\--projected angle/speed block of its
> Hessian is positive definite (minimum eigenvalue $\lambda_{\min}=0.1265$ on every fold), and the
> discrete supply identity \eqref{eq:supply} holds to $\leq5\times10^{-9}$. All five folds pass every
> gate. These are local audits of the quantities named: they do not numerically certify hidden-state
> curvature, discrete or global dissipation, a region of attraction, or any lower bound on escape
> energy (the hidden sector is treated only through its reciprocal-passive construction).

### E7 §V-D2 "三态规则"(行 469–494)
标题 "Critical energy and the three-state margin rule" → "**Candidate critical energy and the
integration-free screen**"。
- eq:critical 前导句(471–472)"…the critical energy is estimated as" → "…the \emph{candidate}
  critical energy is the lowest energy \emph{found} by local search among";公式本体与 eq:margin 不变。
- verdict 三段(484–491)改文:
> Because the local search does not prove that the lowest barrier was found or that every escape
> route crosses one of these configurations, $H_c$ is not a verified lower bound on the true escape
> energy; the margin is therefore a screen, and we issue no formal stability certificate
> (Sec.~\ref{sec:disc} boundaries). The screen has three states:
>  * \textbf{candidate search completed and $M>0$} $\Rightarrow$ screened stable;
>  * \textbf{candidate search completed and $M\leq0$} $\Rightarrow$ screened unstable;
>  * \textbf{candidate search not completed} (no finite candidate) $\Rightarrow$ \emph{refused} (no verdict).
> The refusal direction is conservative: near-critical events whose clearing dynamics escape before a
> saddle can be located are never labelled stable.

### E8 Results 契约(行 ~504–507)
> 原文:…safety metrics are the three-state rule's missed-instability rate (…) and stable coverage
> (fraction of truly stable events **certified stable**).
改文:…safety metrics are the screen's missed-instability rate (fraction of truly unstable events
> screened stable) and stable coverage (fraction of truly stable events screened stable).

### E9 Safety 小节(标题行 653–654;正文 656–679)
标题 → "Integration-free screening: no truly unstable event screened stable on the full-difficulty
out-of-sample set"。
正文关键改句:
- "audits the three-state margin rule on all 315 frozen-test events" → "audits the candidate-energy
  screen on all 315 frozen-test events";
- "no truly unstable event is declared stable: missed-instability rate is $0/51$" → 保留,后加
  "(the 51 truly unstable events span 9 independent source/fault-line groups, and no group contains
  a screened-stable event)";
- "41 are correctly declared unstable and 10 refused" → "41 are screened unstable and 10 refused";
- "The certificates pass on all five folds." → "The local structure audits pass on all five folds.";
- "into a conservative, integration-free verdict" → "into a conservative, integration-free screen";
- 674 "18 of 292 certified events" → "18 of the 292 completed-search events";
- 677–679 fig 引用句 "every certified-stable verdict lies above…" → "every screened-stable event
  lies above…";"no unstable event lies there" → "no truly unstable event lies there";
- 段末加一句(test-reuse 限制):"Because the screen and its zero-miss audit share the same single
  frozen-test read, these rates describe these 315 events and their 9 groups; they are a
  retrospective screen result, not a prospective guarantee."

### E10 表 tab:safety(行 681–703)
caption(682–683)→ "OOS Candidate-Energy Screen (frozen test, all 315 events) and Local Structure
Audits"。
行(690)"Certified events 292 & 292" → "Candidate search completed & 292 & 292"。
行(693)"Missed instabilities 0/51" → "Missed instabilities & 0/51 & 0/51"(保留;组口径在正文/
脚注,见 E9)。
多列脚注(698–701)→ "Local structure audits (all five folds): storage centered 0;
$\|\nabla H(\delta^\star)\|\leq3\times10^{-10}$; COI angle/speed Hessian $\lambda_{\min}=0.1265>0$;
supply identity $\leq5\times10^{-9}$; all pass. The margin is a heuristic screen; no verified global
escape-energy lower bound exists."

### E11 图 fig3 caption(行 707–714)
"Integration-free margin rule over all 315 frozen-test events: candidate critical energy versus
cleared-state storage. … separates **screened-stable** events (above) from **screened-unstable** and
refused events (below). **Screened-stable** events (green) number 61; conservative screen rejections
(amber) and refused events (triangles) dominate the near-critical band. No truly unstable event lies
in the screened-stable region." —— **应用后必须核对 PDF 内文字与 caption 一致**(fig3 由
`experiments/paper_figures.py` 生成,该脚本已在 8db48be 改过;若 PDF 未重生成/仍写 certified,需
重生成或在 caption 中不引入与图面冲突的词)。

### E12 讨论 "Why the margin rule is safe in practice"(行 788–798)
全文改述(标题句改为 "What the screen does and does not establish."),见报告正文建议版:
> Because $H$ is non-increasing under the ZOH deployment up to dissipation, cleared storage below a
> \emph{verified} lower bound on the well's escape energy would imply bounded motion; the local
> candidate search provides no such lower bound, so the margin is a screen rather than a
> bounded-motion guarantee. What is empirical is strong: on events the trajectory fits cannot
> resolve (severe, near-critical), the screen refuses or calls unstable rather than guessing stable,
> so no truly unstable event was screened stable on the full-difficulty set (0/51; 0 of 9 independent
> groups; exact one-sided 95\% upper bound 0.28 on the group failure probability), robust to the
> search-angle perturbation we audited. The price is coverage, conservative by design. A formal
> certificate would require a verified global escape-energy lower bound or a valid BCU-type
> construction; raising coverage into the near-critical band while keeping zero observed misses is
> open---the natural fix (a bounded-trajectory re-check of refused events) reintroduces integration
> and would abandon the integration-free selling point.

### E13 Boundaries(行 800–810)
- item (2)(804–805)"…certified-stable is a sufficient condition, not a complete classifier." →
  "…screened-stable is a heuristic positive call, not a certificate and not a necessary condition;
  the screen is not a complete classifier."
- 追加 safety 泛化边界(并入 items):"The zero-miss audit is retrospective on the single frozen-test
  read and spans only 9 independent groups; it is a descriptive result for this corpus, not a
  prospective guarantee."

### E14 Related-work positioning(行 ~814)
"…we discover which energy-compatible modules are needed and certify the result numerically." →
"…and audit the resulting energy structure numerically."

### E15 Conclusion(行 829–853)
- 834–836: "…the fitted model has a certified storage function and an integration-free three-state
  stability verdict" → "…the fitted model has an audited energy structure and an integration-free
  candidate-energy screen";
- 841–842: "its margin rule gave zero missed instabilities on all 315 full-difficulty events" →
  "its candidate-energy screen never declared a truly unstable event stable on all 315
  full-difficulty events (0/51, over 9 independent groups)";
- 848–851: "the integration-free rule is a conservative sufficient condition with limited stable
  coverage" → "the integration-free screen is conservative and heuristic---a screen, not a
  certificate---with limited stable coverage";
- 851: "a hybrid rule that re-checks refused near-critical events" → "a hybrid screen that
  re-checks refused near-critical events"(措辞随行,可选)。

---

## 2. 中文稿 `paper/main_CN.md`(逐条镜像)

### C1 §3 标题与储能句(行 56–60)
标题(56)→ "## 3. 能量结构审计与免积分候选能量筛查"。
行 60 "清除储能低于最低逃逸能量则轨有界、判稳" → "H 沿轨非增仍由结构与供给恒等式审计支持;
  但『清除储能<最低逃逸能量 ⇒ 有界』需**验证过的全局下界**;局部搜索只给候选,故只作筛查、不发正式证书。"

### C2 证书块(行 62–63)
"**证书(5 折全 PASS)**:储能中心化 0;梯度 ≤3e-10;COI+hidden 投影 Hessian 最小特征值 0.1265>0;
离散供给恒等式 ≤5e-9。不声称全局耗散/全域吸引域。"
→ "**本地结构审计(5 折全 PASS)**:储能中心化 0;梯度 ≤3e-10;COI(角度/速度)投影 Hessian 最小
  特征值 0.1265>0;离散供给恒等式 ≤5e-9。仅审上述量:hidden 曲率/全局耗散/全域吸引域/逃逸能
  下界均未证(不称证书)。"

### C3 三态规则(行 65–67)
"certified & margin>0 ⇒ stable;…未证书 ⇒ refused。临界能量=…能量下确"
→ "**三态筛查(margin=候选临界能量−清除储能,免积分,非证书)**:候选搜索完成且 margin>0 ⇒
  screen-stable;搜索完成且 margin≤0 ⇒ screen-unstable;搜索未完成 ⇒ refused。候选临界能量=局部
  搜索到的相邻鞍点与 ±η=±θ_max 角边界能量最小值;局部搜索不证全局下界 → 不发正式证书。"

### C4 主张②(行 95–110)
标题(95)"免积分判稳:OOS 全难度零漏报" → "免积分筛查:OOS 全难度无非稳被判稳(描述性)"。
表(97–104):"获证书 292/292" → "候选搜索完成 292/292";"漏报不稳定 0/51" → "漏报(无非稳被判稳)
  0/51(=9 独立组)";"stable coverage 0.231/0.299" 保留。
句(106–109)后追加:"9 个独立 source+fault_line 组单侧 95% 精确(Clopper–Pearson)上界 0.28 →
  证据是『本样本零观测』而非『零风险/安全保证』;零漏报审计与 test 同一次冻结读取 → 描述性,
  非对未来扰动的前瞻保证。"

### C5 基线表 modular 行(行 130–133)
列头/单元格:"有(证书+margin)" → "有(本地审计+筛查)";M0 行 "能量结构" 保留(注明仅结构无筛查)。
机制句(139)若含"给判据"措辞 → "既外推又给(启发式)筛查判据"(如原文已含可不动)。

### C6 诚实边界(行 141–150)
item 2(145–146)"判据覆盖有限…OOS stable coverage 0.231(0.9π 0.299);…" → 前缀改"筛查覆盖
  有限(非证书、非必要判据)";并追加边界 item:"安全量为描述性回顾审计:0/51 仅 9 个独立组,单侧
  95% 上界 0.28;与 test 同一次冻结读取 → 不构成对未来扰动的前瞻保证。"

### C7 建议排版(行 152–161)
"表 4 OOS 安全(π/2 与 0.9π + 证书)" → "表 4 OOS 筛查(π/2 与 0.9π + 本地审计)";图 3 注"临界能量
  vs 清除储能" → "候选临界能量 vs 清除储能"。

---

## 3. 应用与验证(单一 coherent 变更,门控执行)

1. 前提:工作树需干净(当前被并发主循环在途 modal.py 修改占用;等其落库或按你授权处理)。
2. 应用 E1–E15 + C1–C7(纯散文/标签,不动数字、不动公式编号)。
3. 验证:
   - `latexmk -pdf`(或 pdflatex×2)编译至 **8 页、0 overfull、无 undefined ref**;
   - 图面文字与 fig3 caption 一致(必要时重生成 PDF);tab:safety/fig:margin 交叉引用保留;
   - `paper/main_CN.md` 与英文稿逐条镜像核对;
   - 跑 G1/G2/G6(tex 变更理论上不影响,仍复跑确认);grep 不再出现 "certified/certificate" 的
     稳定性主张残留(除"screen/certificate 明示不提供"的边界句)。
4. 建议 commit 信息风格:`docs: manuscript re-scoped to candidate-energy screen (per 0002)`。
5. 反证/回退:任一编译错误或漏改清单项 → 回退留 `reports/failed/`。

## 4. 待你定夺(仅两处;其余按 STATUS/0002 机械执行)

- **D1 标题**:建议把 "Integration-Free Stability Margins" 弱化为含 "Screen/Screening" 的措辞
  (E 稿行 35 与 CN 标题)。保守默认改;若你希望保留 "Margins" 卖点也可只改摘要/正文(标题语义由
  摘要"screen, not certificate"句兜住)。
- **D2 摘要长度**:诚实句使摘要变长;需按 IEEE 篇幅压缩(可删 mild/moderate RMSE 部分或频率句)。
- 授权方式:说"按方案执行"即在干净树应用全部;或调整 D1/D2 后执行。
