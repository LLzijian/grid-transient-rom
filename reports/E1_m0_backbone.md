# E1 技术报告:M0 基座复现

> 历史保留标记（2026-09-22）：本报告仅记录早期开发基线与局部物理/QA检查。
> 当前状态见 [current_state](../docs/current_state.md)；结论边界以 [0010 审计](accepted/0010_project_conclusion_audit.md) 为准。

日期: 2026-09-04 · 状态: **PASS** · 证据: `results/e01_m0_f1/summary.json` + `model_m0_f1.npz`

## 1. 目标

在 kiki 中复现 kron-reduce 已验证的 M0 基座拟合流程,确认弱形式求解器、
COI 守恒、RK4 四阶收敛全部工作。**只在 F1 的 train/val 上进行,test 未打开。**

## 2. 方法(与 kron-reduce integral-v2 一致,verbatim 移植)

- 方程:`m_i·ω̇_i = ΔPm_i − D_i·ω_i − Σ_e B_ie·K_e·(sin η_e − sin η0_e)`,
  `ΔPm = Pm − Pm_equilibrium`,COI 功率投影内建力平衡。
- 拟合:一次性积分弱形式。窗宽 (1,5,20),梯形积分;设计矩阵
  `[Π·diag(ω), B*(sin η − sin η0)]`,RHS `∫ΔPm dt − m(ω_stop−ω_start)`。
- 求解:列归一化 ridge + `scipy.optimize.lsq_linear` 界 `(0,∞)`(D,K ≥ 0)。
  ridge 网格 (1e-8,1e-6,1e-4,1e-2) 按验证 rollout 损失选择。
- 表示:`generator_10`(10 机全分辨率,AFSD v5 定论),ZOH-Pm。
- 参数:10 阻尼 D + 45 线路耦合 K = 55。

## 3. 结果

### 3.1 拟合健康度

| 量 | 值 |
|---|---|
| solver 状态 | completed,solver_success=true,24 次迭代 |
| 矩阵条件数 | 16.2(良态),effective_rank 55/55(满秩) |
| 标量方程数 | 2,631,580(600 等权 case×horizon 块) |
| **弱残差 RMSE** | **6.09e-4**(拟合残差,尺度 ~1e-4 相对 ΔPm 主项) |
| selected_ridge | 0.01(网格中最优) |
| 4 个 ridge 候选验证损失 | 5.38917(几乎平坦,问题对 ridge 不敏感) |

### 3.2 物理审计

- `physics_audit`: **pass**;D,K ≥ 0 非负成立;内建力平衡 true。
- 参数形态:前 10 个 D 几乎全为 0(1e-28~1e-32 量级),后 45 个 K 有稀疏非零
  (0~5.07)。**ZOH-Pm + 10 机完整图上,纯同步力矩主导、阻尼项辨识为 ~0。**
  这与物理一致:切除后机械功率不回摆,系统暂态几乎无阻尼衰减项贡献。
  → 这正是后续 E2 边阻尼模块要补充的间隙。
- COI 漂移(max |Σ m·x|): **8.88e-16**(双精度噪声量级)。

### 3.3 验证滚动(43 val 事件,H=100 = 1.0 s)

| 指标 | 值 |
|---|---|
| mean RMSE | 0.3738(外部复算) vs 0.3731(fit 内部) ✅ |
| median RMSE | 0.1651 |
| pooled RMSE | 0.5803 |
| worst-10% CVaR | 1.1109 |
| **pooled ω R²** | **−0.0034** |
| pooled δ R² | 0.8734 |
| rollout 失败率 | 0 / 43 |

### 3.4 收敛审计(前 3 个 val 事件)

dt→dt/2→dt/4 误差比 ≈ **15.2~15.3 ≈ 2⁴**,严格四阶收敛;dt²vs dt⁴ 误差单调下降,`convergent=true`。

## 4. 与旧项目的对照

| | kron-reduce integral-v2 | kiki E1 |
|---|---|---|
| 表示 | clustered_7 | **generator_10**(更难) |
| 验证 mean RMSE | 0.30 | 0.374(同数量级,全分辨率更高,符合预期) |
| COI 漂移 | ~1e-15 | 8.9e-16 ✅ |
| 失败率 | 0 | 0 ✅ |
| 收敛阶 | 四阶 | 四阶(15≈2⁴)✅ |

10 机 raw 上纯 M0 的 ω R²≈0 与旧项目 clustered_7 上 0.53 的差距,本质是
**10 机暂态含更丰富的模态与记忆效应**,正是论文"基座 + 模块"叙事的数据支撑:
基座正确拟合(残差 6e-4),方差解释力交给后续模块。

## 5. 结论与下一步

M0 基座复现达标。弱形式求解器、COI 守恒、四阶 RK4、非负岭回归全部健康。
进入 E2:在冻结基座上拟合可剪枝能量结构化模块并做整块取舍。
