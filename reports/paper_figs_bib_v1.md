# 论文成稿技术报告:真图 + 真实参考文献(v1.1 增量)

> 历史快照标记（2026-09-22）：本文档仅记录早期图表/文献节点，不代表当前投稿就绪或证据等级。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；权威边界见 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 与 [0010 审计](accepted/0010_project_conclusion_audit.md)。

日期: 2026-09-04 · 状态: **完成** —— 英文稿 `paper/main.tex` 升级到 v1.1:
①三张论文数据图从冻结模型/不可变结果生成并接入;②参考文献从"内嵌 10 条经典"升级为
bibtex + `paper/bib/references.bib`(旧项目既有真实条目 9 条 verbatim + 经典条目);
③全稿用 TeX Live **编译通过(8 页 PDF,无 undefined citation/ref,唯一 benign balance
warning)**。中文简稿 `paper/main_CN.md` 同步更新。

## 1. 图生成(experiments/paper_figures.py,复用旧项目 venv)

| 图 | 内容 | 依据(全部冻结后/不可变) | 数据源 |
|---|---|---|---|
| fig1_pipeline | 模块化识别流程示意(纯示意,非数据) | 方法规程(§III) | — |
| fig2_difficulty_trajectories | 分难度代表轨迹:truth vs modular,F1 折 **validation** 事件,10 机 COI 速度,H100=1 s ZOH | 每 band 取 ω-RMSE 中位事件 | `e02_prune/model_pruned.npz`(冻结)+ F1 val 一次性 rollout |
| fig3_margin_scatter | 免积分 margin 全景:315 test 事件,临界能量 vs 清除储能,判稳/判不稳/拒 vs truth | 逐事件 margin(冻结模型已产出) | `results/e06/{F1..F5}/test_margin.json`(只读聚合,不再读轨迹) |

- **纪律核对**:fig2 只用 F1 **val**(选择集,开发用),test 未因出图而重读;
  fig3 直接读 E6c 已产出的不可变 test_margin 聚合——不触碰原始 test 轨迹、不改任何模型。
- fig3 图例计数与主表逐项一致:certified 292、safe 61、保守拒 190、不稳定判不稳 41、
  missed 0、refused stable 13 / unstable 10(脚本核验打印)。
- 输出到 `paper/figures/*.pdf`(正文引用)+ `*.png`(仅本地核验用,不接 tex)。
- 复现: `E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe experiments/paper_figures.py`
  (加载模型 ~2 s + F1 43 val rollout + 画图,全程 <1 min)。

## 2. 参考文献(真实、可核验)

- 9 条自 kron-reduce 既有 `references.bib` verbatim 找回:Brunton 2016 / Messenger 2021
  Weak-SINDy / Loiseau 2018 constrained Galerkin / Sarić 2021(电力符号回归,TPWS)/
  Fiaz 2013(port-Hamiltonian 电网)/ Guo 2018 / Sharafutdinov 2018 / Thümler 2022(按原
  arXiv 形态)/ Cui 2021(ANDES 混合符号-数值框架)。
- 补经典:kundur1994、chow2013coherency、bergen1981、pai1989、chiang2011、
  athay1979、vanderschaft2017、raissi2019、hochreiter1997、schmid2010。
- 新增正文 \cite 位置:绪论 SINDy 段(weak-SINDy/Galerkin/电力符号回归)、能量段
  (P-H 电网)、§II 数据(ANDES 生成语料 + IEEE39 出处)、§III-D 弱式(Weak-SINDy 定位)。
- 未引但保留于 bib:guo2018/sharafutdinov2018/thumler2022(bibtex 会自动只列被引项)。
- **仍需作者投稿前逐条核对卷/期/页**(尤其经典条目),见缺口 §4。

## 3. 编译验证

`cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` → exit 0,
8 页 PDF,`main.bbl` 自动生成;无 undefined citation/reference,唯一警告为 balance 包
"called \balance in second column"(IEEE 惯用无害提示)。

## 4. 遗留缺口(作者侧)

1. **作者块/资助/单位**:仍是模板占位。
2. **参考文献细节**:投稿前对照出版商记录逐条核验卷期页;如需增补近年电力-SINDy/降阶/
   Lyapunov 直接法文献,提供 .bib 条目即可内嵌。
3. **图样式**:fig2 面板视觉密度与配色可在你审阅后按期刊偏好微调(脚本可复现);
   fig3 若想突出 0.9π 敏感性可加第二面板(数据在 `results/e08/summary.json`)。
4. 数字抽查:建议抽查任一表格/图例数字对回 `results/`(图例已程序化自 results 读取)。
