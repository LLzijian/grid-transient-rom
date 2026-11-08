# 论文 LaTeX 排版质量报告(overfull 清零节点 v1)

> **历史归档 · 2026-09-22**：历史过程记录：仅记录旧版论文的排版检查，不能证明当前版本内容或投稿就绪。
> 原文保留供追溯。有效入口：[论文状态](../../paper/STATUS.md)；[后续结论审计](../accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-04 · 状态: **完成** —— `paper/main.tex`(v1.1b)排版质量审计与修复:
overfull hbox 8 处 → **0**,overfull vbox 0,页数维持 **8**,exit 0,无 undefined
cite/ref。仅余 benign underfull(两栏排版的正常松排噪声)。

## 1. 审计发现(修复前)
编译日志 8 处 overfull hbox,全部是**内容超出 IEEE 单栏列宽**(252 pt),投稿可见的缺陷
(溢入栏间隙/对栏):

| # | 量 | 内容 | 类型 |
|---|---|---|---|
| 1 | 14 pt | eq:backbone | 单行摆动方程过宽 |
| 2 | 22 pt | eq:nodeho | 单行含 3 子句 |
| 3 | 85 pt | eq:resmem | 单行含 4 子句 |
| 4 | 130 pt | eq:resimpulse | 长积分式 |
| 5 | 100 pt | eq:storage | 双 underbrace 累加式 |
| 6 | 163 pt | eq:critical | 长 set-builder 文本 |
| 7 | 28 pt | tab:select | 7 列宽表(单栏) |
| 8 | 94 pt | tab:baselines | 5 列表格+长表头(单栏) |

## 2. 修复(全部逐 token 保留,不改任何数学/数字/文字内容)
- **6 个显示式 → amsmath 多行**:eq:backbone/eq:nodeho/eq:resmem 用 `aligned`(每行
  `&=` 对齐、短行容纳);eq:resimpulse/eq:storage 用 `aligned` 续行对齐
  (`={}&`/`&{}-\int` 保运算间距),eq:storage 进一步把 kinetic / backbone-potential /
  ΣU 三 underbrace 分三行;eq:critical 改 `multline`(首行左、次行右,`\Bigl\{…\Bigr\}`
  固定尺寸跨行合法)。标签全部保留 → 正文 `\eqref{eq:backbone}`、`\eqref{eq:supply}`
  解析不变。
- **tab:select(仍单栏)**:表体 `\footnotesize` + `\tabcolsep 4pt` → 28 pt 溢出消除。
- **tab:baselines**:74 pt 残溢单栏无法收纳(8 pt 下仍 ~326 pt 自然宽),转 **全宽
  `table*`(跨两栏顶部 float,文字保真不删表头)** → 溢出消除,8 页不变。

## 3. 修复后编译核验
- overfull hbox **0** / overfull vbox **0**;Pages **8**;exit 0;
- 无 undefined citation / reference、无 multiply defined;
- 残余:7 underfull hbox(badness 1418–10000,Intro 段落最甚)+ 2 underfull vbox
  (badness 10000,@\output)= 两栏常规松排,无观感风险,不值得为此引入新 overfull。

## 4. 给作者的视觉核对点(本模型无法预览 PDF)
1. **tab:baselines 现为顶部通栏 float** —— 请确认其落页位置合理(无突兀大留白);
   若想回归单栏并删减表头措辞(如 "LSTM (autoregressive)"→"LSTM"),告诉我即可。
2. 6 个公式折行后的对齐观感;eq:critical 的 multline 换行点是否顺眼。
3. 图 2/图 3 本身仍待你人眼 QA(见 paper_figs_bib_v1 §4)。

## 5. 复现
`cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;核查:
`grep -c "Overfull .hbox" main.log` → 0。
