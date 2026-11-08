# 技术报告:main.tex 诚实化改写(并发在途)核验 —— 已完成区正确,Discussion/Conclusion/§release/bib 未完成

> **历史归档 · 2026-09-22**：历史过程记录：这是已结束的改写中间态，所列“未完成”项目不能作为当前待办。
> 原文保留供追溯。有效入口：[0004 完整改写验收](../accepted/0004_manuscript_honesty_rescope.md)；[论文状态](../../paper/STATUS.md)。

日期: 2026-09-05 · 状态: **只读核验(在途改写的中间态)** —— 另一作者正在改 `paper/main.tex`
(工作树未提交,HEAD `4ce8565`)。本报告不代改、不代提交;列出"已完成且正确"与"未完成缺口"。

## 1. 结论速览

- 在途改写覆盖 **标题→§Results-§pm**(§V 起),方向与 accepted 0002/0003、`paper/STATUS.md` 及
  我的两份方案(`manuscript_screen_rewrite_plan_v1.md`、`evidence_governance_0003_manuscript_v2.md`)
  **完全一致**,且更彻底:
  - 标题 → "…Energy-Structured Reduced Models with Integration-Free Candidate Screening";
  - 摘要改为探索性口径(0/51 over **9 groups**、单侧 95% 精确上界 0.283、被消费 build、**无
    confirmatory 声明**、pre-registered blocked release);
  - "single read of frozen test" 假声明已删;315 全部标 retrospective/exploratory;
  - 模块选择节承认 F4 legacy 候选过不了修正总门、不能建跨折稳定性;基线标 development/exploratory;
  - modal ~7% headline **整体撤回**(等修正全对协议 + 门内最终模型,现在不报数)——比我的方案更严;
  - §V-D:本地审计 + "candidate critical energy" 三态筛查,引 Chiang/BCU、CP 引文。
- **编译实测:8 页、0 overfull、bibtex 后 chiang1987foundations/chiang1994bcu 均解析。**
- **未完成缺口(请补齐)**:
  1. `\label{sec:release}` 未定义,却被 241 与 500 行 `\ref` → 需新增 pre-registered release 节;
  2. `\cite{clopper1934}`(657 行)缺 bib 条目(`paper/bib/references.bib` 无 clopper1934);
  3. **Discussion(761–812)与 Conclusion(813+)仍是旧文**,残留与新口径冲突的声明:
     774 "certificate at all";776–778 "Because H is non-increasing … lowest escape energy …
     implies bounded motion"(我 v1 E12 判为必须重写的那句定理);792 "certified-stable is a…";
     807 "structurally certified";823 "certified storage function and … three-state stability
     verdict";829 "its margin rule gave zero missed";831 "frequency within ~7%"(legacy best-pair
     值,0003 已判非确认)。
  4. `paper/main_CN.md` 尚未镜像(工作树仅 main.tex 被改)。

## 2. 逐段核验(已完成区,未发现残留过度声明)

对工作树 main.tex 做关键词扫描,剩余 "certificate/certified" 命中仅在**诚实否定句**
("…required for a stability certificate"、"not a stability certificate"、"…are therefore not
retained")与 Discussion/Conclusion 旧文(见上缺口 3)。其余("single read"、best-pair、per-event
best、declared stable/unstable、zero missed guarantee)在已完成区已清除。

## 3. 建议(交由在途作者/下一 loop,不代改)

1. 补 Discussion/Conclusion 改述:可用我的 v1 E12/E13/E15 与 v2 C-a…C-e 逐句替换(在途作者未引,
   但语义与它一致);结论里删 "~7% / margin rule zero missed",改 "候选能量筛查在 315 被消费 build
   上无非稳被判稳(0/51,9 组,CP 0.28),探索性;modal 修正协议待 fresh holdout"。
2. 新增 §release(可引 `configs/test_release_v1.json` 语义:污染 build id、claim-before-read、
   single-pass、sweep 禁令、fresh build 前置),并给 `\label{sec:release}`。
3. `references.bib` 补 `clopper1934`(Clopper & Pearson 1934, Biometrika 26(3-4):404-413)。
4. 完成后 `latexmk -pdf` 应 0 undefined reference/citation;再镜像 `main_CN.md`。
5. 门控:main.tex 变更建议以单一 coherent commit 走 0002/0003 同款验收(改述须与已接受代码/证据
   语义一致 = 天然 G5 claim 对齐;G1/G2/G6 复跑确认无旁效)。

## 4. 边界

本报告只读,不构成对在途改写的验收;工作树被在途作者占据,main.tex 我不再触碰以避免冲突。
权威依据:`reports/accepted/0002…`、`reports/accepted/0003…`、`configs/test_release_v1.json`、
`paper/STATUS.md`、`results/README.md`。核验命令:paper/ 下 `latexmk -pdf -interaction=nonstopmode
main.tex`;行号以当前工作树 `paper/main.tex` 为准。
