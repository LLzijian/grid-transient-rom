# 论文成稿技术报告:英文 IEEE 稿 + 中文简稿(v1)

> 历史快照标记（2026-09-22）：本文档仅记录早期成稿节点，不代表当前投稿就绪或证据等级。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；改写后的权威边界见 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 与 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-04 · 状态: **成稿完成(v1)** · 产物:
`paper/main.tex`(英文,IEEE TPWS 模板,内嵌参考文献,可直接 pdflatex)
与 `paper/main_CN.md`(中文简稿) · 用途: 对接用户最新指令"以 paper/main.tex 为模板、
英文用 /nature-writing 撰写、同时生成中文简稿;旧正文不作数、绪论可参考"。

## 1. 关键决策(与本项目纪律一致)

- **内容全部来自 kiki 真实语料**(E0–E8 + E5b/E5c/E9),非旧版 7-area/M2a/M2A-LYAP 结果。
  `main.tex` 旧正文(§Algorithm Design 起)整体作废,未再引用;新稿所有数字可回溯不可变
  `results/` 与 `reports/synthesis_paper_main.md`。
- **paper_type = algorithmic**(nature-writing 轴定稿):论文章节走
  `任务+范围 → 系统是什么 → 设计原理 → 评估 → 消融隔离贡献 → 失效/边界 → 适用域`。
- **术语表锁定**(全文唯一称谓):M0 swing backbone / 三族 energy-compatible physics
  modules / causal ZOH-Pm contract / clearing state / module inclusion = block pruning
  (source-stratified 2% 非劣门)/ storage function H / energy certificate / three-state
  integration-free margin rule / difficulty bands(mild/moderate/severe)/ metrics
  (missed-instability rate, stable coverage)。中文简稿用同一套翻译,避免漂移。
- **可编译性**:旧 `\bibliography{bib/references,bib/misc}` 指向的文件不存在
  (kiki/paper 仅 main.tex),改内嵌 `thebibliography`(10 条经典文献),文件可独立编译。
- **免图**:现有语料未生成论文图;`.tex` 顶部注释列了 3 张待绘图(管线图、分难度轨迹面板、
  margin 示意图),正文不引用不存在的 `\includegraphics`。

## 2. 主张 → 证据 → 状态(claim-evidence map)

| 主张(稿件内) | 证据 | 状态 |
|---|---|---|
| ① 模块化模型在有效域 OOS 精确再现暂态 | E6b 一次性 test 分难度表: mild n=69 δ0.013/ω0.069、moderate 0.046/0.190、severe 0.378/1.231 | supported(test 冻结后只读一次) |
| ①b 模块选择跨折稳定、增益来自模块且难-条件 | E2/E6 val 跨折:ωR² 基座→模块 F1 −0.02→0.702…F3 0.442→0.802;5 折全保留 3 族;val 难度中位 0.37–2.24 | supported(模块增益限温和/中等 val) |
| ② 免积分三态判稳全难度 OOS 零漏报 | E6c pooled missed 0/51、coverage 0.231;E8 0.9π coverage 0.299、判决集逐事件不变;5 折证书(center 0、grad ≤3e-10、Hessian λmin 0.1265、supply ≤5e-9)PASS | supported(经验审计;稳健性=非定理) |
| ③ 外推频率/阻尼 | E5b pooled ~7.2%(186 事件)、单折 4.0–10.7%;E5c 550 对 model ζ 0.100 vs 0.094、|Δζ| 0.040;MAC 弱为数据侧 | supported(振型不纳入主张) |
| ④ Pm 自有方程 = 视界依赖门控 | E3 H100 拒绝(−2.1%)/ E3b H800 采纳(+35.3%,τ≈2.0 s,89 系数) | supported |
| ⑤ 无能量基线既输精度也无判据 | E7 F1: modular 0.045 vs M0 0.071/PINN 0.124/LSTM 0.346/SINDYc 发散(36/43 非有限);PINN 点态 0.122 闭环反逊 M0 | supported(限 F1 折、CPU 预算) |

## 3. 结构说明

- 引言重写:沿用旧稿"可参考"的领域脉络(高维模型→降阶/直接法→SINDy→能量方法→图结构
  稀疏辨识),但导向本框架、给出四贡献;不再复述旧 7-area 错误叙事。
- 主表与合成报告 §3 建议一致:表 1 轨迹(分难度)、表 2 模块选择(带难度列)、表 3 基线、
  表 4 安全(π/2 与 0.9π + 证书)、表 5 Pm 门控。
- 诚实边界在 Discussion 与 Conclusion 各出现一次,方向安全,不越证据。

## 4. 需要你后续核对的缺口(作者侧)

1. **参考文献**:10 条为经典文献的高置信条目,但**逐条 bibliographic 细节请在投稿前
   对照原 bib/官方源核对**(尤其卷期页码)。如你认为应增补近年 power-SINDy / 降阶 /
   Lyapunov 直接法文献,把 .bib 条目给我即可嵌入。
2. **作者块 / 资助 / 单位**:仍为模板占位(First A. Author 等),需你替换。
3. **图表**:3 张待绘图未生成;若有数据图代码可接入 `figures/`,我再在正文插入引用与
   编号。
4. **数字复核**:稿内全部数字从不可变结果抽取,但建议你抽查任意一格(尤其 baseline 表中
   PINN/LSTM 与表 4 的 0.9π 行)对回 `results/`。

## 5. 如何重定向

- 只改错处:点名哪一节/哪一句,我只改那一处,其余保持(见 nature-writing workflow §9)。
- 若需要把正文切回你更熟悉的某位作者口吻,给一小段样例我按其句式/术语校稿。
