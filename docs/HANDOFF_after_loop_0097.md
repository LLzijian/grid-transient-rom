# Agent 交接：Loop 0097 后的研究状态

交接日期：2026-09-22。实验检查点：`0c8eb2f`。分支：`codex/task2-long-horizon-campaign`。

## 1. 先确认权限与状态

**项目已按用户要求在 Loop 0097 完成后暂停。此次请求只是编写交接文档，不是恢复实验。**

- 0097 已关闭，无待收尾实验；0098 尚未预注册、未实现、未启动。
- 总体研究目标尚未完成；暂停不是 Goal success，也不是科学 blocker。
- 接手 agent 可以阅读和核对交接材料；必须收到用户明确恢复指令后，才能启动下一科学 loop。
- 最新授权研究方案是 [TASK2_after_loop_75.md](TASK2_after_loop_75.md)，不是 after66 或更早方案。新指令若更新方案，以新指令为准，但不得追改旧实验判定。
- 交接收尾时工作树新出现未跟踪的 `docs/TASK2_after_loop_97.md`。本次没有读取、修改或启用它；接手时先核对用户是否已明确指定其替代旧路线。文件出现本身不解除暂停。
- 保留当前工作树中的其他用户改动；不要使用 `git reset --hard`、批量还原或清理命令来制造“干净仓库”。

### 最短恢复阅读顺序

1. [AGENTS.md](../AGENTS.md)：root/worker 权限、独立挑战、暂停与停止条件。
2. [current_state.md](current_state.md) 和 [engineering_loop.md](engineering_loop.md)：当前导航与执行协议。
3. [TASK2_after_loop_75.md](TASK2_after_loop_75.md)：当前科学路线及 L1/L2/L3 交付边界。
4. [0097 预注册](../reports/task2_complex_balance_0097.md) 和 [0097 结论](../reports/accepted/0097_complex_balance.md)。
5. 只有在确定下一个问题时，再读下表对应报告与必要代码；不要从 Loop 0001 重读整个历史。

这份交接是导航和解释，不替代原始预注册、不可变证据或用户后续指令。

## 2. 当前究竟接受了什么

| 对象 | 路径 | 身份与限制 |
|---|---|---|
| M3-Pm 参考 | `private_data/task2_m3_0020/model.npz` | 保留的物理参考，未覆盖 |
| 短期 0058 | `private_data/task2_joint_0058/model.pt` | 保留的短期参考，未覆盖 |
| 长期 S | `private_data/task2_long_0060_5s/model.pt` | 暂定长期参考，未覆盖；不是新方案的最终成功模型 |
| 原始 G3000 | `private_data/task2_structure_0078_extend_G-fixed/step3000/` | 当前诊断基点，不是已正式升级的主模型 |
| 0093 / 0096 candidate | 各自实验目录下的 `candidate/` | 已拒绝的独立候选，只保留证据，不得改作默认模型 |

**没有新的最终 L1/L2/L3/M4 模型升级。** `reports/accepted/0097_*` 中的 accepted 只表示“诊断假设获支持”，不表示模型通过开发或冻结确认。

路线目标仍是同一套 100 ms 初始化器与显式自治 ODE，支持暂态预测、声明域内的稳定性分析及可信物理模态。能量模板恒等式、局部稳定极点、拟合误差下降，不能互相替代这些能力。

## 3. 到当前结论的最短证据链

| Loop | 已获得的信息 | 不允许据此声称 |
|---|---|---|
| 0078 | C3000 保留轨迹性能但只匹配 1/13 个 FOM 模态；G-fixed3000/G-auto1200 未通过轨迹保留 | 非凸优化充分、结构模型族被否定 |
| 0079–0088 | 多种条件回放及稳定性约束修正未通过联合部署门槛；0088 风险比 .974324，未达 .90 | 可继续只加训练资源救回旧分支；其累计预算已关闭 |
| 0089–0091 | 原始 G3000 和 0088 各 0/13 模态；谐波反作用误差 1.04415；仿射 LOO capture .784977 未达 .80 | 局部稳定足以建立模态可信度 |
| 0092 | 现有 26 个势能系数及 10 个阻尼系数方向捕获 .686483616 的实频反作用差异 | 这组系数部署后必然改善轨迹 |
| 0093 | 精确部署 0092 系数：4/13 模态；自主平均风险比 3.19129，严重事件均值/最差四例比 5.64575/6.66372，拒绝 | 正系数与局部稳定能够保障自主预测 |
| 0094 | 在同一模型 q 回放、真实物理轨迹条件下，0093 修正使平均积分平衡风险变为 1.54803 倍，严重事件均值 4.46777 倍；双步长支持 | 条件回放等于自主预测，或 q 是隐藏真值 |
| 0095 | 一次凸兼容性拟合返回 `optimal_inaccurate / AlmostSolved`，故按原门槛判未决；返回点 capture .571587392，条件约束的直接检查通过 | 0095 已通过，或该点是获证最优解 |
| 0096 | 精确部署 0095 返回点：局部稳定，但 0/13 模态；自主平均风险比 2.52959，严重均值/最差四例比 3.19500/4.75107，拒绝 | 条件非劣足以保障自主非劣 |
| 0097 | 0096 在实际复 FOM 极点与物理振型方向上，13/13 平衡缺陷 eta>.10，满足预注册的至少 7/13；数值与独立复算通过 | 自主误差已被因果归因、r=3 不足、整个模型族失败 |

近期报告入口：

- [0092](../reports/accepted/0092_backbone_span.md)、[0093](../reports/failed/0093_backbone_deployment.md)、[0094](../reports/accepted/0094_conditional_balance.md)。
- [0095](../reports/failed/0095_backbone_compatibility.md)、[0096](../reports/failed/0096_witness_deployment.md)、[0097](../reports/accepted/0097_complex_balance.md)。
- 各轮完整公式、门槛、预算、命令和哈希以对应 `reports/task2_*_00NN.md` 与结论报告为准。

## 4. Loop 0097：接手时最重要的结果与算法

这是 **diagnostic / supported / promotion none**。没有训练、仿真、新的特征值求解或 TEST 读取。

固定三套 33 维外部坐标 Jacobian：G3000、0093、0096。对每套计算相同 13 个缓存 FOM 复极点及对应振型。

```text
外部状态：[delta(10), omega_rad_s(10), p(10), q(3)]，h=(p,q)
H = lambda I13 - A_hh
B = A_hdelta + lambda A_homega
X = solve(H, B)
T = diag(M) [A_omega,delta + lambda A_omega,omega + A_omega,h X]
R = diag(M) lambda^2 - T
L = diag(Sdelta / (M Somega)), K = L R, a = v_delta / Sdelta
eta = ||K a|| / (||L diag(M) lambda^2 a|| + ||L T a||)
```

先用 `[COI delta/Sdelta, Omega*omega_pu/Somega]` 的范数归一化完整 FOM 特征向量。**作用 R 时保留全部原始 delta 分量，不做 COI 投影。** 非零 lambda 下共同 delta 与真实共同频率相连，投影会改变被检验的模态。

- 0096 eta 范围 `.135375593–.946746623`，13/13 大于 .10；G3000 为 13/13，0093 为 12/13。
- 最大隐藏系统均衡化条件数 `6.03710`，求解工程误差指标 rho 最大 `1.74266e-14`，不是近奇异求解导致的假信号。
- 独立复算最大数组相对差 `6.04e-15`、标量绝对差 `5.33e-15`。
- 主程序 39 次小规模线性求解、78 次 SVD；独立程序 39 次原始线性求解，未独立重算 SVD。
- 主运行 `3.077982 s`，独立 `1.6495104 s`，加元数据预算 `10 s` 共 `14.7274924 s`；阶段已关闭，无真实重试。
- 完整逐模态 eta、分母、绝对缺陷、K 奇异谱、隐藏求解尺度与残差均已保存，不必重新计算。

结论仅是：当前候选的实频矩阵拟合收益未消除实际复模态方向的平衡缺陷。eta 不是极点距离，也不是旧频率/阻尼/振型匹配门槛的替代品。

## 5. 必须保持的物理与数据语义

- `r=3`；GRU 只由过去 100 ms 的 11 点角度/转速历史初始化 q；当前阶段仍用真实物理 p0。自主预测中不得重新读未来真值。
- `p = GENROU.tm - pm_star`，不是 `tm*omega`。`omega = (omega_pu-1)*Omega`，`Omega=2*pi*60`；`M=raw_inertia/Omega`。
- 内部结构坐标为 `[delta, pi=M*omega, p, q]`。能量/J/R/B 运算与外部 omega Jacobian 之间必须做正确链式变换。
- 仅移除共同角度得到 32 维分析空间；保留共同频率。不强迫所有发电机反馈零和，不把电气转矩与忽略铜损的 Pe 混同。
- 当前 26 个角差势能系数 k 与 10 个阻尼 d 只直接改变 omega 行；本次固定结构中它们不直接改变 p/q 方程。自主运行中 q 仍会通过变化的物理轨迹间接受影响。
- `StructuredROM._pos = softplus(raw)*mask`，没有自动叠加 `1e-8` 底值；0095 约束底值不等于模型参数化底值。
- 固定尺度：`Sdelta=.47675314588683515`，`Somega=1.0033667249602924`；p 的逐机尺度从既有缓存/归一化文件读取，不重新拟合。
- 不恢复 whitening 或 latent covariance 正交性作为状态独立性的主要验收门；不据本轮失败增加 latent dimension、库或损失。
- 699 维 FOM descriptor 含代数/PSS 约束，不能把所有非零时间常数状态直接当成已合法消元的最小 ODE。

**数据隔离：**本分支自主评价缓存含 346 个合格 TRAIN 事件（311 stable、35 unstable；原始 369，24 故障组）。保留真实长度和 NaN 尾部，不能制造完整 5 s 真值或删除严重事件。边界 0075/0077 数据未合并。开发 VAL 不是最终独立确认。新扩展/补充 TEST 仍冻结；旧 Pilot TEST 曾在历史评估中消费，永久排除后续选择用途。

旧部署筛选门槛便于理解，但不得自动替下一轮完成预注册：

- 平均 1/2/5 s 风险为 `.8*mean + .2*worst35`，候选/原基线需 `<=.90`；严重 35 例均值与最差 4 例风险比均 `<=1.05`。
- 原基线平均/严重均值/最差四例为 `.8938195532229601 / .6654646977500597 / 2.303873238746948`。
- 物理模态一对一匹配：频率相对误差 `<=.15`、阻尼绝对差 `<=.05`、MAC `>=.80`，部署筛选要求至少 7/13。
- 局部 alpha `<=-.01/s`、FD/特征值敏感性及 RK4 `.005/.01` 双步长数值门槛独立检查。局部稳定不能代替轨迹、模态或区域性证书。

## 6. 代码与证据地图

以下路径均相对于仓库根；`private_data` 是本地证据，不要假设 fresh clone 自带。

| 用途 | 文件/目录 |
|---|---|
| 结构模型与原语 | `experiments/structured_rom.py`、`structure_grammar.py`、`train_structured_rom.py` |
| 0097 主诊断与测试 | `experiments/audit_complex_balance_0097.py`、`tests/test_complex_balance_0097.py` |
| 复数求解 QA | `experiments/harmonic_solve_0090.py`、`tests/test_harmonic_solve_0090.py` |
| 0097 结果 | `private_data/task2_complex_balance_0097/{arrays.npz,evidence.json}` |
| 0097 独立复核 | `private_data/task2_complex_balance_0097_independent/{check_complex.py,independent.json}` |
| 0096 原始/候选 Jacobian 与部署证据 | `private_data/task2_witness_deployment_0096/` |
| 0093 部署/自主预测复用代码 | `experiments/audit_backbone_deployment_0093.py`、`backbone_forecast_0093.py` |
| 0095 单次凸拟合与 Gram 辅助代码 | `experiments/audit_backbone_compatibility_0095.py`、`backbone_compatibility_0095.py` |
| 0095 返回系数、双网格 Grams/factors、求解状态 | `private_data/task2_backbone_compatibility_0095/{arrays.npz,evidence.json}` |
| 0094 固定 q 条件回放与积分证据 | `private_data/task2_conditional_balance_0094/{arrays.npz,metrics.npz,evidence.json}` |
| 0092 36 列实频投影与系数 | `private_data/task2_backbone_span_0092/{arrays.npz,evidence.json}` |
| 0090 FOM/ROM 反作用缓存 | `private_data/task2_harmonic_reaction_0090/{arrays.npz,evidence.json}` |
| 0078 对齐的 FOM 复极点/振型 | `private_data/task2_control_modes_0078/modal_arrays.npz` |
| TRAIN 自主基线/真值/qenc 缓存 | `private_data/task2_guarded_rollout_0088/trajectories.npz` |

重要读取细节：

- 0078 必须用同一 `target_recomputed_indices` 索引 `descriptor_eigenvalues` 与 `descriptor_right_vectors`；目标 ID 为 `62,64,66,68,71,73,77,79,84,91,98,100,104`。
- 0078 缓存向量用于 0097 复平衡；旧模态匹配使用已核对的 0035b 目标频率/阻尼元数据，不要混为同一套标签。
- 0090：`A_F` 为 699x699，`E_F` 为长度 699 的对角向量，`A_G` 为 33x33。谱范数 `||diag(E_F)||2=max(abs(E_F))`，不是向量的欧氏范数。
- 0097：`baseline_A33/failed0093_A33/witness0096_A33`；每个 arm 有 `H/B/X/T/R/K`、完整均衡化求解记录与作用向量。JSON `per_target` 共 39 行。
- 0095：`beta` 36 维，`k0/d0`；0092 基值键却叫 `coupling0/damping0`。勿根据命名猜测。
- `source_paths` 与前后 SHA 保存完整继承链。老键 `code`、`code93` 等有既定身份，新增脚本用新键，不能覆盖旧来源。不要修改已被旧证据哈希绑定的辅助代码来救回旧实验。
- 条件路径及其 Grams 使用未来物理真值，只能作条件诊断/训练约束，不能包装成因果预测。

## 7. 环境、可安全核对的入口和 Git

```powershell
Set-Location 'E:\03-nn_Transfer_Func\kiki'
$researchPython = 'E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe'
git status --short
git log -6 --oneline
Get-Content docs/current_state.md
Get-FileHash private_data/task2_complex_balance_0097/arrays.npz
```

PowerShell 解析含 `D/d` 等大小写不同键的 JSON 时用 `ConvertFrom-Json -AsHashtable`。不要为这次交接安装包或重建环境。

已完成的合成测试入口（仅供之后修改代码时使用，不要求接手就重跑）：

```powershell
& $researchPython -m pytest tests/test_complex_balance_0097.py tests/test_harmonic_solve_0090.py -q
& $researchPython private_data/task2_complex_balance_0097_independent/check_complex.py --self-test
```

真实 0097 执行命令已记录在结论报告中，**不要因为接手而重跑**；现有目录拒绝覆盖，且阶段预算已关闭。新机器缺少原始缓存时先报告缺失，不能重新生成一份冒充旧证据。

若恢复后科学计划确实需要凸求解：既有 CVXPY/Clarabel 位于 `.loop-work/research_deps_0067`；先导入主 venv 的 NumPy/SciPy，再按 0095 脚本方式加入该目录，避免替换主数值运行库。0095 原 solver 的 status 门槛仍不可放宽。

提交锚点：`bc223cd` 预注册，`86b5aa3` 实现与合成检查，`0c8eb2f` 结论和暂停。交接文档提交在其后，不改变科学检查点。

当前工作树另有用户/外部文档修改，包括 `AGENTS.md`、`README.md`、`docs/engineering_loop.md`、大量报告/稿件，以及历史报告删除、`reports/history/`、`.tmp_long_smoke/`、`loop97.zip` 等未跟踪项。接手时重新看 `git status`，不要清理、恢复、提交这些不属于自己任务的改动。仅显式 stage 自己的文件。

## 8. 证据锚点与迁移要求

交接时已读文件字节核对的模型 SHA256（G3000 仅为诊断基点）：

```text
M3    4928f40e54656fa05a5d478969f4d53f63e7551a824f761312efc793bc0621b4
0058  d82b1d8d75528eb7e55c28f5f5438270fd95eca2499b5afec5486b6894545f67
S     d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c
G3000 2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b
```

M3/S/G3000 与 0097 继承记录一致；0058 不在 0097 的直接来源表中，此处单列其既有模型哈希，勿声称它经过了 0097 部署验证。

0097 SHA256：

```text
arrays.npz       8626c3ec9977561eb7c3a19e16111dea853fe93a830f109d8379c54d6ce8199f
evidence.json    6c8dbd27516678027eed258fbeef5aeb07f9a88cd05d7a6b3beec9374eeff8ad
independent.json 6c88e476a446db6b5ff3afe0a6e224a5d69155c588f04fd63424cfd8555717a3
check_complex.py aa5e9c3b8caa5cb8df07556cdbefc80d9220fe86df3b91de9e168c29a16b3c6e
```

Git 代码不是完整科学快照。跨机器交接还必须保留所用 `private_data` 模型、数组、JSON、独立 checker，以及继承链引用的工件；需要凸求解时还需既有依赖目录或经过核验的等价环境。不要改写旧 JSON 中的绝对路径来伪造原始哈希一致；另记路径映射、核对原始字节。没有这些工件只能阅读报告，不能声称完整复现。

## 9. 用户恢复后，最小下一步是什么

**以下只是待 root 决策的问题，不是预注册，不是暂停期间的执行清单。**

可以检验：现有同一组 36 个物理 k/d 方向，能否在固定基线权重及 TRAIN 条件兼容性约束下，减少实际复模态方向的平衡缺陷？这针对“实频矩阵目标与复模态方向目标不一致”，而非通过扩维救回旧候选。

若 root 选择此方向：

1. 先读 0095 的确切凸约束与未决原因、0096 的自主拒绝、0097 的复平衡结论。确认固定隐藏子系统下 `Delta R = D^T diag(Delta k*cos(phi)) D + lambda diag(Delta d)` 的符号、单位及适用条件。
2. 在看到任何新拟合结果前，另立有限预算、分支数、有效性与验收门槛；不能沿用旧 loop 名义追加预算或放松原门槛。
3. 如用复方向残差作拟合目标，归一化权重需固定于基线，不能允许参数依赖分母把 eta 做小；条件非劣仍不代表自主非劣。
4. 任何新系数都必须另经实际部署的数值、局部、物理模态及自主轨迹/严重事件检查；FOM 监督过的模态成绩是开发证据，不是独立最终确认。
5. 若换成其他科学方向，由 root 解释证据与理由；在重大路线变更前取得独立只读挑战并记录处置。不许 worker 自主选择科学假设或改物理结构。

恢复前的最小外部条件只有：**用户明确要求继续**。当前没有待监控进程，不需要用自动化或后台任务突破暂停。
