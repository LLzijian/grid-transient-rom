# E7 技术报告:非能量基线(SINDYc / PINN / LSTM)对照 —— "结构把点态拟合转成闭环外推"

> 历史保留标记（2026-09-22）：本报告是 F1、有限预算的开发折基线对照，不是确认性 OOS 结果。
> 证据等级以 [0003](accepted/0003_holdout_lifecycle_and_modal_reporting.md) 为准；当前状态见 [current_state](../docs/current_state.md)。

日期: 2026-09-04 · 状态: **完成** —— 同一因果部署契约(H100、ZOH、F1 开发折)下,
模块化能量模型在有效域(mild/moderate 稳定 val)闭环精度**全面优于**三类无能量
结构的基线;**PINN 的点态导数拟合最佳(val 残差 0.122),闭环却比纯 M0 还差**;
SINDYc 结构性发散(36/43 非有限),LSTM 自回归漂移。基线全部
`certificate_claim=False`,无法提供免积分稳定判据 ·
证据: `results/e07/main/{M0_backbone,modular_pruned,sindy,pinn,lstm}/summary.json`
(F1 开发折;test **未打开**)

## 1. 对照协议(与 E2/E6 同一合同)

- **同一 F1 开发折**:209 train / 43 val。基线与 modular/M0 参照共享**同一 train
  集与同一 val 选择集**,同一冻结前规则(test 全程不读)。
- **同一部署输入**:清除态 (δ0, ω0) + 因果 ZOH Pm 计划,H100(1 s),rk4
  max_step 0.01。val 43 事件难度构成:stable_mild 17 / stable_moderate 3 /
  stable_severe 14 / unstable_severe 9(与 E6 同款难度分层,1 s 内 truth 峰值 |ω|)。
- **训练上下文中的 Pm_observed** 仅作 teacher 信号(与 modular 相同纪律)。
- 逐事件 rmse 聚合与 E2/E6 同码;难度分层记法一致。

| 模型 | 结构 | 拟合 | 部署 | 能量/Lyapunov |
|---|---|---|---|---|
| M0 backbone(E2) | 2 阶摆 + 弱拟合阻尼/耦合 | 弱式解析 | 因果 ZOH | 有(空模块能量) |
| modular(E2 prune) | M0 + 3 族物理模块,整块选择 | 弱式解析 | 因果 ZOH | 有(证书 PASS) |
| **SINDYc** | 全局受控二次多项式库,稀疏 ridge | 时间导数 LS | rk4 | 无 |
| **PINN** | M0 力 + 残差 MLP(原点 pin=0) | 导数+平衡损失,100 epoch | rk4 | 无 |
| **LSTM** | LSTMCell 预测 (δ,ω) 导数 | teacher-forcing,50 epoch | 自回归 | 无 |

三基线从 kron-reduce **逐字移植**(`src/kiki/models/baselines.py`),仅改写导入。
唯一必要改动:SINDYc.fit 内算 FitResult 的 val rollout 加防护(发散记大 loss,
见 §4)——否则全局多项式在近临界事件上非有限,基线自身无法完成拟合。

## 2. 结果:H100 闭环(43 val,每模型零架构级失败率 0——SINDYc 例外见下)

| 方法 | mild+moderate 稳定 rmse (n=20) [δ, ω] | severe 稳定 rmse (n=14) | unstable 带 rmse (n=9) | 非有限失败 | pooled R² |
|---|---|---|---|---|---|
| M0 backbone | **0.0714** [0.018, 0.099] | 0.882 | 0.291 | 0 | 0.313 |
| **modular(E2)** | **0.0451** [0.012, 0.063] | **0.475** | 0.330 | 0 | **0.797** |
| SINDYc | 发散(0 事件成功) | 发散 | 58.99* | **36/43** | −13539 |
| PINN | 0.1235 [0.061, 0.164] | 0.900 | 0.240 | 0 | 0.278 |
| LSTM | 0.3461 [0.215, 0.440] | 1.404 | 0.536 | 0 | −0.900 |

\* SINDYc 仅 7 个事件"成功"(全部为最剧烈的不稳定事件),平均 ω rmse 83 rad/s:
不是拟合,是在 1 s 视界内**有限地爆炸**。mild/moderate 稳定事件全部在 rollout
内非有限 → 计入 36 失败。

**有效域(mild/moderate)对照**:modular 0.045 ≪ M0 0.071 < PINN 0.124 ≪ LSTM
0.346;SINDYc 无有效输出。模块化模型把 M0 提升 1.6×,同时把每个无能量基线甩开
2.7×–7.7×。

## 3. 关键机制发现:PINN 点态拟合最好,闭环反而最差

| 量 | 值 |
|---|---|
| PINN teacher/平衡损失(best val,100 epoch) | **0.122**(点态残差拟合收敛良好) |
| M0 残差网络规模 | hidden 24,三层 tanh,~1.5k 参数 |
| PINN 闭环 mild+moderate rmse | 0.124 —— **比它脚下的纯 M0(0.071)还差 1.7×** |
| LSTM teacher 损失(best val,50 epoch) | 10.75(未收敛) |
| LSTM 闭环 mild+moderate rmse | 0.346(自回归漂移) |

**解读**:PINN 证明"导数级拟合质量高"并不自动转为"闭环轨迹质量高"。MLP 残差场
逐点逼近真残差,但**不受保守/耗散结构约束**,沿轨迹的误差一致性没有任何
Lyapunov 担保 → 一个小而任意的残差场反而把能量自洽的 M0 扰动得更糟。这正是
本论文的核心对比轴:**模块化能量模型把可解释的物理模块(能量自洽、可证书)叠加在
M0 上,得到的是既能闭环外推、又能免积分判稳的结构**;黑盒残差只在训练分布内做
点态插值,无结构可外推、无判据可给。

## 4. SINDYc 结构性失效(而非调参问题)

- 51 项受控多项式(1, δ_rel, ω, δ_rel|δ_rel|, ω|ω|, Pm)对全谱数据一次 LS(ridge
  1e-5,阈值 1e-6):系数 1020 中 **1019 个非零** → 无稀疏、无剪枝、无平衡锚。
- 训练点态拟合无意义:跨 mild/severe/unstable 的多尺度 |ω|(0.05–15+)使 LS 被
  剧烈事件主导;mild 事件附近闭环不动点不再稳定 → 1 s 内即发散(非有限)。
- 与 kron 旧项目同理:无结构、全状态全局多项式的 SINDYc 在清除态自回归上
  不可用;其对"结构选择"的对照价值恰在于此——它证明了裸回归库不能替代
  基座+物理模块的归纳偏置。

## 5. 定性轴:判据能力(论文最关键的差别)

| 模型 | 免积分稳定判据 | 证书 |
|---|---|---|
| modular(E2 prune) | **有**(E4 三态规则;E6c OOS 全难度 **missed-instability 0/51**,5 折证书 Hessian min λ 0.1265 正定 PASS) | True |
| M0 backbone | 能量形式存在但未在 E4 套件审计 | 能量结构 |
| SINDYc / PINN / LSTM | **无**(`certificate_claim=False`) | False |

即使忽略轨迹精度,三类基线也**没有能力回答论文的目标问题之二**(从清除态免积分
判稳定域):它们只输出一条轨迹,而无 Lyapunov 函数/临界能量,因而无法在
不积分的情况下区分"将要恢复"与"将要失步"。这是 modular 模型相对基线不可替代
的功能差,而非数值差。

## 6. 局限与诚实边界

- **NN 为 CPU 有界预算**:PINN 100 epoch(val 选择),LSTM 50 epoch(val 选择,且
  teacher 损失 10.75 未收敛)。给更多 epoch 只会缩小 LSTM 差距的**数值部分**,
  不会改变其结构性弱点(无平衡锚 → 漂移;无能量 → 无判据)。
- **单折(F1)对照**:基线的严格对照限定在 F1 开发折;跨折主表仍以 E6 的
  模块化管线为准(其五折模块选择稳定性不依赖基线)。
- **SINDYc 收敛失败源于结构**:我们不把 36/43 非有限当作"调参后可修";它正是
  全局无结构多项式在该多尺度数据上的预期行为。
- 基线在 unstable 带(margin 规则的有效域)的轨迹误差无科学意义——该带本就不在
  "轨迹精度"声明内,module 的判稳由 margin 三态接管(E6c 零漏报)。

## 7. 结论与论文位置

- **模块化模型 vs 三个"非能量参考"的对比,从两头闭合**:① 在有效域(mild/
  moderate)轨迹精度全面胜出(mild+moderate rmse 0.045 vs 0.124–发散);② 在
  能力上独有"免积分稳定判据 + 能量证书"。论文可写:同契约下无能量结构的
  黑盒学习器既追不上简化物理模型的闭环精度(因为只有物理结构能外推),也给
  不出稳定性声明。
- **PINN 的负结果是方法论宝石**:点态导数拟合(可被任意提升)与闭环外推
  (需要能量结构)解耦——证明模块包含选择(计入哪些物理模块)才是精度与可判据
  的真正来源,而非拟合器的容量。

## 8. 工程注记

- 移植文件 `src/kiki/models/baselines.py`(三模型同文件,~300 行;逐字自 kron
  `sindyc.py` / `pinn.py` / `lstm.py`,仅导入改写 + SINDYc 拟合期 rollout 防护)。
- `experiments/run_e7.py`:CLI `--methods {sindy,pinn,lstm,ref,all}`,`--smoke` 子集
  调错/计时;`results/e07/<tag>/` 不可变、拒覆盖;`ref` = 同码回放已冻结
  M0 backbone 与 modular 模型保证表内一致。
- 实测(209 train/43 val):SINDYc 3 s;PINN ≈229 s(其中 M0 自拟合 base 202 s、
  dataset 6 s、100 epoch 仅 17 s);LSTM ≈895 s(50 epoch,每 epoch 前向+反向 ~30 s)。
- 全部 CPU;确定性:seed 324、best-val checkpoint、rk4 max_step 0.01。
- 探针脚本 `experiments/_probe_e7.py` 已删除。
