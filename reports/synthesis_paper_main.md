# 综合收官报告:论文结果主表与主张—证据装配(E0–E7)

> 历史快照标记（2026-09-22）：本文档保留为历史数字装配记录，不代表当前投稿就绪或当前 claim 口径。
> 当前状态见 [paper/STATUS.md](../paper/STATUS.md)；已完成改写与边界以 [accepted 0004](accepted/0004_manuscript_honesty_rescope.md) 和 [0010 审计](accepted/0010_project_conclusion_audit.md) 为准。

日期: 2026-09-04 · 状态: **完成(历史装配记录,claim 表述已被 2026-09-05 取代;表格仍为数字来源)**

> ## ⚠️ 证据等级与 claim 口径更新(2026-09-05,supersedes 本文 claim 表述)
>
> 本文是 2026-09-04 的**历史装配记录**,早于 accepted loops 0002/0003/0004。其**表格是仍有效的
> 数字来源**,但其 **claim 表述须按以下口径重读**:
> - 局部 SLSQP/root 搜索**不证全局逃逸能下界** → 文中"证书 / 判稳 / 零漏报"一律改读为
>   **本地能量结构审计 + 候选能量筛查**(正式证书状态恒 refused,非证书)。
> - 315 语料已被 E6b/E6c/E8 等**重开读取** → 0/51 等为**描述性/回顾性**(51 非稳事件仅
>   **9 个独立 source/fault-line 组**,单侧 95% 精确上界 **0.28**),**非 confirmatory**。
> - **confirmatory OOS 结果当前不存在**:需全新 dataset build + 修正总门 refit 后由预注册单次
>   释放(`configs/test_release_v1.json`,blocked)给出;后验角度/阈值扫描在 holdout 上被禁。
> - 权威口径:`reports/accepted/0002_candidate_energy_not_certificate.md`、
>   `0003_holdout_lifecycle_and_modal_reporting.md`、`0004_manuscript_honesty_rescope.md`、
>   `paper/STATUS.md`、`results/README.md`。逐条改述见手稿 `paper/main.tex` / `paper/main_CN.md`(v2)。
>
> 原状态行的"test 在模型冻结前从未打开、本次综合不改动任何结果"仅作历史自述;自 0003 起该
> test 语料已被判定为已消费,结论不再成立(见上)。

## 0. 一句话结论

一个以 **M0 二阶摆为永不删除基座**、**模块包含由数据门控**(物理先验模块 +
Pm 自有方程)的**模块化能量模型**,在中轻扰动清除后暂态上 OOS 精确再现
(mild δ_rmse 0.013)、把纯 M0 的 ω R² 从 −0.02 抬到 0.70–0.80(温和/中等折)、
以能量/Lyapunov 结构提供**免积分稳定判据(全难度 OOS 零漏报)**、并外推预测
振荡频率(单事件中位 ~4% 误差);同一契约下无能量结构的基线(SINDYc/PINN/LSTM)
既追不上其闭环精度,也给不出任何稳定性声明。

## 1. 数据与协议(每份主表的共同前提)

- IEEE39 规范集 315 事件(264 stable / 51 unstable),`combined_line_5fold_v1`
  线路不相交五折;每折 train≈206–214 / val≈37–47 / test 61–66。
- 因果 ZOH-Pm 契约:Pm_observed 仅作训练 teacher,部署自 Pm_clear 常数保持;
  rollout H100(1 s)/RK4 max_step 0.01;COI 坐标,effective mass。
- **难度条件化**贯穿主表:按清除后 1 s 内 truth 峰值 |ω| 分 mild(≤0.5)/
  moderate(0.5–1.5)/severe(>1.5,近临界)。E6 确立了轨迹精度**难度条件化而非
  折条件化**,且 val 是温和开发集、test 覆盖全难度——故论文不给单一折
  mean_rmse,而给分难度表。

## 2. 论文主张 → 证据主表

### 主张①:简化模型准确反映系统暂态(模块包含 = 模型选择)

**A. 跨折验证(选择一致性)**——prune 整块后向删除,三族模块在 **5 折全部保留**,
模块选择跨折稳定。val ω R²(M0 → modular):

| fold | n_val | backbone ωR² | full/prune ωR² | prune 保留 |
|---|---|---|---|---|
| F1 | 43 | −0.02 | **0.702** | 全 3 族 |
| F2 | 37 | 0.595 | **0.748** | 全 3 族 |
| F3 | 42 | 0.442 | **0.802** | 全 3 族 |
| F4(severe val) | 43 | 0.459 | −0.031 | 全 3 族 |
| F5(severe val) | 47 | −3.99 | −2.77 | 全 3 族 |

温和/中等折(F1–F3)模块化一致抬升 ω R²(Δ+0.15~+0.72);F4/F5 的 val 近临界占
多数——F1–F5 稳定 val 中位 max|ω| = **0.50/0.37/0.88/1.32/2.24** rad/s
(`results/e09_valdifficulty/summary.json`;F2 实际 0.37,为全折最温和)——增益消失
是该难度带超出降阶模型域,温和事件上同模型依然精确(见 B)。

**B. OOS 分难度轨迹(test,一次性;论文主表核心)**:

| 难度 | n(稳定) | δ_rmse | ω_rmse |
|---|---|---|---|
| mild ≤0.5 | 69 | **0.013** | **0.069** |
| moderate 0.5–1.5 | 40 | **0.046** | **0.190** |
| severe >1.5 | 155 | 0.378 | 1.231 |
| ALL stable | 264 | 0.290 | 0.947 |

**C. 基线对照(同契约 H100,F1 val)**——见 §4。

### 主张②:Lyapunov/能量形式保留 → 免积分判稳(从清除态,无需暂态积分)

三态规则:证书 & margin>0 ⇒ stable;证书 & margin≤0 ⇒ unstable;未证书 ⇒
**refused(绝不判稳)**。margin = 临界能量(切除后鞍点搜索,angle_limit=π/2)−
清除态储能,**免积分**。

**D. 证书(5 折)**——energy storage 中心化 0、梯度 ≤3e-10、投影 Hessian 最小
特征值 **0.1265**(正定,PASS,五折一致)、供给恒等式误差 ≤5e-9。

**E. margin 三态审计(全部 315 test,OOS 全难度)**:

| 指标 | val(E4) | OOS test(E6c) | OOS 0.9π(E8) |
|---|---|---|---|
| 不稳定漏报判稳 | **0/9** | **0/51** | **0/51**(逐事件不变) |
| 不稳定判 unstable | — | 41/51 | 41/51(不变) |
| 不稳定 refused | — | 10/51 | 10/51(不变) |
| stable safe coverage | 0.50 | 0.231 | **0.299** |
| 全难度含近临界带 | 否(温和 val) | **是(零漏报成立)** | 是(零漏报稳健) |

免积分规则在未训练、含近临界 severe 的事件上把"漏报不稳定"压到 0;代价是
稳定覆盖保守(severe 稳定多被判 unstable/拒绝),方向安全。E8 敏感性审计:放宽
临界能量搜索角至 0.9π 覆盖升到 0.299、**不稳定判 un/refused 集合逐事件不变**,
零漏报非 knife-edge;但 margin **非**严格随角度单调(292 获证书事件中 18 个
下降)——安全稳健性是经验重验结果非定理。操作点维持先验可辩护的 π/2。

### 主张③:预测额外系统特征(振荡频率)

**F. 频率与阻尼外推(E5 + E5b + E5c,冻结模型)**——模型在切除后平衡处线性化
轻阻尼极点预测振荡特征。E5b 跨折兑现 E5 的频率折间承诺:**频率 pooled 中位
~7.2%**(186 个逐事件 OOS val,<10% 占 65%;单折随折难度 4.0%–10.7%,温和 F1
最利、severe F5 最紧;旧项目单折 28%)。E5c 收口阻尼:**阻尼比 pooled 无系统性
偏差**(550 轻阻尼对,model ζ 中位 0.100 vs emp 0.094,偏差 +0.001),|Δζ| 中位
**0.040**(折间 0.029–0.047);E5 的 F1"模型过阻尼"(0.089 vs 0.062)是 F1 温和
val 的单折伪象。MAC 弱(0.39–0.51)为数据侧限制(DMD 4 s 窗内模态弱激励、形状
不收敛;频率特征根稳健),非模型坐标缺陷。**外推属性对 212 个 OOS 事件稳定
成立,但频率精度随难度自然收紧——主表宜引 pooled ~7% 而非 F1 4%。**

### 主张④:Pm 自有方程是一个**模块包含决定**(数据门控,E3/E3b)

**G. 视界依赖结论**:同一因果门(≥2% 验证改进)在 **H100(1 s)拒绝**
(improvement −2.1%,回退精确 ZOH);在 **H800(8 s)采纳**(+35.3%,raw rmse
0.0182→0.0118,因果 Pm-ODE 追回 ZOH→oracle 差距的 74%)。选中谱含 **τ=2.0 s**
慢极点(物理上与 TGOV 调速器 ~2.1 s 一致,≤2.5 s cap);89 非零系数。
方法论意义:Pm 模块计入与否是**经验、视界依赖**的门控决策,非先验。

### 主张⑤(基线,收官对照)

见 §4。

## 3. 论文主表(建议排版)

合并 A/B/E/C 为一张主结果表 + 一张安全表 + 频率一行:
- **表 1 轨迹精度(分难度,不可给单一折 mean_rmse)**:B 的行 + 行间注明
  "val(温和)与 test(全难度)分布差异是属性,非 bug"(各折 val 难度构成见表 A
  注)。
- **表 2 OOS 安全**:E 的 pooled 行(0/51,coverage 0.231)+ D 证书行。
- **表 3 外推与选择**:A(模块跨折稳定)+ F(频率)+ G(Pm 门)。
- 表下统一诚实边界(§5)。

## 4. 基线对照(§2-C 细节)

同一 H100 因果契约、同一 F1 43-val、难度分层与 E6 同码;三基线逐字移植自
旧项目(kron-reduce),`certificate_claim=False`。

| 方法 | mild+moderate rmse | severe rmse | 非有限失败 | pooled R² | 免积分判据 |
|---|---|---|---|---|---|
| M0 backbone | 0.071 | 0.882 | 0 | 0.313 | 能量结构 |
| **modular** | **0.045** | **0.475** | 0 | **0.797** | **有(E4/E6c)** |
| SINDYc | 发散 | 发散 | 36/43 | −13539 | 无 |
| PINN | 0.124 | 0.900 | 0 | 0.278 | 无 |
| LSTM | 0.346 | 1.404 | 0 | −0.90 | 无 |

**机制结论(E7)**:PINN 点态导数拟合最好(val 残差 0.122),闭环却比纯 M0 差
1.7× → 点态拟合质量 ≠ 闭环外推质量;唯有能量结构(物理模块)既外推又判稳。
SINDYc 全局无锚多项式结构性发散(非调参问题)。这从反面闭合了"模块包含才是
精度与可判据来源"。

## 5. 诚实边界(论文须一并呈现)

1. **有效域**:模型对 max|ω|₁s ≲1.5 rad/s(中轻扰动)精确;近临界 severe 事件
   是降阶模型固有边界——误差方向保守(发散/过判失步),不构成漏报。
2. **判据覆盖有限**:OOS stable coverage 0.231(0.9π 时 0.299,见 E8);未获证书
   的稳定事件多为近临界深动态。已审计角度敏感性(E8:0.9π 零漏报不变);大幅
   提升需混合判据(margin 拒绝的近临界事件转有界轨迹复核)——这会重新引入
   积分、超出"免积分判稳"卖点,留作未来工作。
3. **val/test 难度不均**:开发折温和,主表必须给分难度,不能给单值。
4. **频率/阻尼**:折间稳定性已由 E5b/E5c 兑现(频率 pooled ~7.2%,单折
   4.0–10.7% 随难度;阻尼 |Δζ| 中位 ~0.04、pooled 无系统性偏差);振型(MAC)
   未达成(数据侧限制)。
5. **基线预算**:NN 为 CPU 有界 epoch(PINN 100 / LSTM 50,后者未收敛),单折;
   更多 epoch 只缩数值差距,不改变其结构性弱点。

## 6. 主张—报告—产物追溯

| 主张 | 分报告 | 不可变产物 |
|---|---|---|
| ① 轨迹精度/模块选择 | E2, E6, E7 | `results/e02_*`、`results/e06/{F2..F5}/*/summary.json`、各折 `test/summary.json`、`results/e07/main/*/summary.json`、`results/e09_valdifficulty/summary.json` |
| ② 免积分判稳 | E4, E6c, E8 | `results/e04/summary.json`、`results/e06/margin_oos_summary.json`、`results/e06/{F1..F5}/test_margin.json`、`results/e08/summary.json` |
| ③ 频率/阻尼外推(折间) | E5, E5b, E5c | `results/e05/summary.json`、`results/e05b/summary.json`、`results/e05c/summary.json` |
| ④ Pm 模块门控 | E3, E3b | `results/e03/{summary,checkpoints/}`、`results/e03b/summary.json` |
| ⑤ 基线对照 | E7 | `results/e07/main/...` |
| 数据/管线 | E0, E1 | `results/e00_data_check/summary.json`、`results/e01_m0_f1/{model_m0_f1,summary}.json`、`src/kiki/` |

## 7. 再现命令

```bash
PY=E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe   # 复用旧项目环境
$PY experiments/run_e6.py F2 backbone   # 逐折:backbone/full/prune(每折同一三阶段)
$PY experiments/run_e6.py F2 full
$PY experiments/run_e6.py F2 prune 100
$PY experiments/run_e6b_test.py            # 全部冻结后一次性 OOS test 读取
$PY experiments/run_e6c.py                 # OOS margin 三态审计(全难度)
$PY experiments/run_e8.py --angles 0.9     # margin angle_limit 敏感性(≈2.3 h)
$PY experiments/run_e5b.py                 # 频率外推折间确认(全折 OOS)
$PY experiments/run_e5c.py                 # 阻尼比折间确认(同批配对)
$PY experiments/run_e3.py                  # H100 Pm 门(三段链)
$PY experiments/run_e3b.py                 # H800 Pm 门
$PY experiments/run_e7.py --methods all --pinn-epochs 100 --lstm-epochs 50
```
所有输出目录拒覆盖;test 只在冻结后读一次。
