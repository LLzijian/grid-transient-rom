# E3 技术报告:机械功率自有方程(Pm-ODE)门控实验(F1)

> 历史保留标记（2026-09-22）：本报告仅作为视界依赖的早期 Pm 开发诊断记录。
> 其继承结论已按 [accepted 0005](accepted/0005_manuscript_scientific_consistency.md) 收窄；当前状态见 [current_state](../docs/current_state.md)。

日期: 2026-09-04 · 状态: **视界依赖的模块包含决策 —— H100(1 s)拒绝、H800(8 s)采纳(Pm 方程改进 35.3%)** ·
证据: `results/e03/{summary.json, checkpoints/}`(H100 门)+ `results/e03b/summary.json`(H800 门) ·
F1 折,209 train / 43 val,test 未打开

## 1. 方法(M0-GMD-PM 三段链,逐段 physics audit)

```
m03eg : TransientEmfM0Model  (3 阶 M0 + 节点瞬态 EMF)
gmd   : GraphModalDiscoveryM0Model (冻结 m03eg 上的低秩图闭合)
pm    : CausalPmGraphModalM0Model = M0-GMD + stable GraphModalPmDynamics
```

Pm 端口方程定义在零总功率子空间(1ᵀp=0):`a = U_pᵀ(Pm−Pm_eq)`,
`v = U_pᵀ Π_power(ω)`,`ȧ = A a + B v`,A 经谱平移约束 **Hurwitz + τ≤2.5 s**
(ANDES TGOV1N T3≈2.1 s 上限)。部署因果:自 Pm_clear 初始化推进,精确
`A=B=0`(ZOH)为显式候选。门:**H100 验证 ≥2% 改进且零失败率**否则保留 ZOH。

## 2. 三段结果

| 段 | 方法 | fit | best_val_loss | audit | 备注 |
|---|---|---|---|---|---|
| m03eg | M0-3E | completed | 2.3110 | **pass** | 瞬态 EMF backbone |
| gmd | M0-GMD | completed | 2.3110 | **pass** | 冻结 backbone,零闭合被选(无增益) |
| pm | **M0-GMD-PM** | completed | — | **pass** | **promoted=False**, improvement **−2.1%** |

Pm 段最终选择:`Pm_clear_ZOH`(精确回退),0 系数。
- `physics_audit.status=pass`:`selected_is_exact_ZOH_fallback=true`,
  causal_inputs_only、Pm_clear 精确初值、零总功率不变式(Σbasis~1e-16)均成立;
  Hurwitz 项按"exact ZOH 或 Hurwitz"规则通过。
- **弱系统诊断(209 train, dense 151 / highorder 58)**:零总功率 Pm 端口
  低秩结构确实存在——响应奇异值分散,99% 有效秩 **6**,非可忽略。结构可辨识,
  但独立预测未胜过 ZOH。

## 3. 关键数值:H100(1 s)三种策略

同一 M0-GMD backbone,部署策略不同(全 43 val,零失败):

| 策略 | 总 mean_rmse | highorder(20) | dense(23) | 因果? |
|---|---|---|---|---|
| ZOH(Pm≡Pm_clear) | 0.2478 | 0.0252 | 0.4413 | 是(基线) |
| **Pm-ODE(门控后=ZOH)** | **0.2478**(与 ZOH 恒同) | 0.0252 | 0.4413 | 是 |
| observed-Pm oracle(线性插值) | 0.2451 | **0.0129** | 0.4470 | **否**(非因果) |

Pm 独立拟合(H100,teacher-forced):最佳 ODE 候选 standalone pm rmse
0.1996(mean)/0.1476(median)/0.5229(p90),不优于 ZOH。

## 4. 解读(负结果为何、以及它在方法论上的意义)

- **在 1 s 视界上,Pm 信息价值微小**:即使**非因果**地已知真实 Pm 计划,
  oracle 总 RMSE 也只改善 1.1%(0.2478→0.2451)——dense(近临界/不稳定)事件
  的误差由摆方程速度动力学主导,而非 Pm 预测误差。因果 Pm ODE 在此视界
  没有可提取的 ≥2% 改进空间 → 门控正确拒绝、保留干净的 ZOH。
- **时间尺度错配假说**:调速器恢复 τ≈2.5 s ≫ 1 s 门。Pm 的自有方程价值应
  出现在若干秒视界(此时 ZOH 把 Pm 恒持在 Pm_clear 的误差随 ~1−e^{−t/τ} 增长,
  而 ODE 可因果恢复该慢趋势)。highorder(稳定、≥18.6 s 轨迹)上 oracle 已在
  1 s 把误差减半(0.0252→0.0129)——但那需要非因果知识;因果 ODE 只能追慢模态。
- **与旧项目对照**:kron-reduce 旧折(178 train/34 val,不同源混合)上同一模型
  promoted(+12.9%,选中 2.5 s 慢极点 + 快极点)。kiki 规范 F1 折(dense 占
  val 过半)在 H100 不成立。门控纪律的价值正在于此:不因旧折正结果而强推,
  在新数据上按证据回退并如实报告。
- 三段链本身(结构发现、Hurwitz 投影、ZOH 候选、逐段审计、因果纪律)全部
  按设计运转;失败的只是"在这个视界这个折上 Pm 方程值得计入"这一科学论断。

## 5. E3b 结果(长视界 H=800,验证时间尺度假说)—— **时间尺度假说证实**

在 highorder 稳定事件(n=20,轨迹 ≥18.6 s)上以 H=800(8 s)重跑 Pm 门控
(复用冻结 M0-GMD backbone;dense 事件因长视界必然发散而按源排除;弱系统
训练仍含全部 209 事件)。因果契约不变,门不变。

| 量 | 值 |
|---|---|
| **promoted** | **True** |
| **validation_improvement_vs_ZOH** | **+35.3%**(raw rmse 0.01825 → 0.01181) |
| 选中候选 | stable_graph_modal_Pm_ODE,ridge 1e-4,thr 0.03,margin 0.5 |
| 谱 | 慢极点 **τ=2.0 s**(abscissa −0.5,恰在 2.5 s 调速器上限内)+ τ 0.96 s、0.34 s + 快极点 |
| 系数 | 89 非零(9 维零总功率模态 × 自/交叉耦合) |
| standalone Pm 预测(H800) | ODE **0.0116** vs ZOH 0.0309(**3 倍改进**) |
| physics audit | **pass**,Hurwitz=True,τ_max=2.0 s ≤ 2.5 s cap,非 ZOH 回退 |

H800 轨迹误差(highorder val,零失败):

| 策略 | mean_rmse | mean_δ | mean_ω | 因果? |
|---|---|---|---|---|
| ZOH | 0.0182 | 0.0070 | 0.0248 | 是(基线) |
| **Pm-ODE(因果)** | **0.0118** | 0.0040 | 0.0162 | 是 |
| observed-Pm oracle | 0.0096 | 0.0037 | 0.0131 | 否 |

**因果 Pm-ODE 追回 ZOH→oracle 差距的 74%**((0.0182−0.0118)/(0.0182−0.0096))。

## 6. 结论与论文意义

- **Pm 确实有可辨识、因果可用的自有方程**:在 ≥ 数秒视界(≳ 2×调速器
  恢复时间常数)上,一个自 Pm_clear 因果初始化、Hurwitz 约束、驱动于预测
  相对速度的端口 ODE,把 8 s 预报误差降低 35%,并追回非因果 oracle 差距的
  约四分之三。2.0 s 慢极点的发现与 TGOV 调速器时间常数(≈2.1 s)物理一致。
- **模块包含是视界依赖的经验决策**——这与论文"模块化选择"哲学完全一致:
  在 1 s 快速评估合同下 ZOH(Pm 常数)已充分;在 8 s 视界下 Pm 记忆效应必须
  计入。两个门都在无 label、因果纪律下用数据决定,而非先验。
- **反观 H100 的负结果**:其价值在于防止"在短视界强推多余模块";H800 的
  正结果则给出模块的必要物理理由。二者共同构成对 Pm 端口的完整模块化
  审计。

## 7. 工程注记

- H100 门总运行时 ~29 min(m03eg 3' / gmd 20' / pm 6');GMD 弱方程网格
  (5 rank×15 τ×5 thr×3 ridge)为最重段;E3b(H800)约 14 min。
- 保存 m03eg/gmd/pm(+ E3b pm_long)及嵌套 backbone 检查点;物理审计逐段执行。
- 所有门的 H100/H800 外部 rollout 均零失败;COI 漂移 ~1e-15。
- E3b 验证集按源(highorder)划分而非按 label;dense 事件因长视界必然发散
  而排除,故 H800 结论限定为"长稳定轨迹上的 Pm 记忆效应"。
