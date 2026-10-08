# 论文 Release-Candidate 交接报告(单页索引)

> 历史快照标记（2026-09-22）：本文档仅记录早期 release-candidate 交接，不代表当前投稿就绪或证据等级。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；权威边界见 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 与 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-04 · 状态: **英文稿可投稿就绪(作者侧动作除外)** · 本文档是四份节点报告
(draft_v1 / figs_bib_v1 / number_audit_v1 / tex_quality_v1)之上的**单一交接索引**。

## 1. 分层就绪状态(v1.1b)

| 层 | 状态 | 核验 | 出处 |
|---|---|---|---|
| 内容/方法(nature-writing, algorithmic) | ✅ | 主张→证据→边界完整,D/E 各一次 | draft_v1 |
| 数据与纪律 | ✅ | 315 语料因果 ZOH、分难度、test 只读一次;图 2 仅 val、图 3 只读不可变聚合 | draft_v1 / figs_bib_v1 |
| 数字审计 | ✅ | 5 大块全 PASS;3 处问题闭环(①基线列头 ②H800 子集披露 ③频率难度方向) | number_audit_v1 |
| 图(3 张,矢量) | ✅ 数据侧 | fig1 示意 · fig2 F1-val 分难度轨迹 · fig3 315-test margin;脚本可复现 | figs_bib_v1 |
| 参考文献 | ✅ 已接线 | 19 条真实(9 条 kron-reduce verbatim + 经典),bibtex 自动只列被引 | figs_bib_v1 |
| 排版/编译 | ✅ | **8 页,0 overfull,无 undefined cite/ref**;tab:baselines 转通栏 float | tex_quality_v1 |
| 中文简稿 | ✅ | main_CN.md 与 EN 逐项同步(含全部修订) | — |

## 2. 产物清单

| 产物 | 路径 |
|---|---|
| 英文稿(IEEE TPWS) | `paper/main.tex` → `paper/main.pdf`(8 页) |
| 中文简稿 | `paper/main_CN.md` |
| 参考文献 | `paper/bib/references.bib`(19 条) |
| 图 | `paper/figures/fig{1,2,3}_*.pdf`(fig2/3 另存 png 仅本地核验) |
| 节点报告 | `reports/paper_draft_v1.md` · `paper_figs_bib_v1.md` · `paper_number_audit_v1.md` · [已归档排版报告](history/paper_tex_quality_v1.md) |
| 事实来源 | `reports/synthesis_paper_main.md`(主张→证据主表) |

## 3. 复现命令与产物级核验
- 图:`E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe experiments/paper_figures.py`(<1 min,只读 val/results)
- 编译:`cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- 门:`grep -c "Overfull .hbox" main.log` → 0;页数 8。
- **渲染层核验(2026-09-04 补)**:`pdftotext -layout main.pdf` 后抽查 16 个审计头条数字
  全部命中(0/51 ×6、0.797、0.231、0.299、7.2、0.045/0.071/0.013/0.069/1.231、
  18 of 292、0.040/0.100/0.094、2.0、89 non-zero)——提交 PDF 的渲染文本与审计源一致,
  无静默漂移。

## 4. 剩余作者侧动作(自动无法代办)
- [ ] **作者块 / 单位 / 资助**(模板占位 `First A. Author`)
- [ ] **参考文献卷期页对照出版商记录**(投稿前逐条核;若增补近年电力-SINDy/降阶文献,给条目即嵌)
- [ ] **图与表格排版人眼 QA**:fig2 面板密度、fig3 配色;tab:baselines 现为顶部通栏 float,确认无突兀留白(想回归单栏我可删减表头措辞)
- [ ] 摘要/关键词/期刊模板元数据最终过目
- [ ] 投稿系统手续

## 5. 结论与建议
`main.tex` 内容、数字、图、参考文献、排版**五层闭环**,无已知未决技术问题;测试纪律
(test 只读一次、标签只审计、不可变 results、因果契约)全程未破。**建议在此自然节点交由你
审 PDF 并做 §4 作者动作**;如需我继续(改图、加引用、调措辞、打包投稿附件),给出具体指令即可。
