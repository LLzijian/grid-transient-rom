# PDD v3.1 独立跨模型族复核报告

---

## 一、审查身份、范围、输入校验与未检事项

- **审查者实际宿主身份**：**Antigravity（Gemini 4 Neon）**，相对本轮 PDD v3.1 的 Codex 团队作者（`gpt-6-astra` 签发规范 + `gpt-6.1-sol` 同步与静态复算）属于**跨模型族独立只读复核者**，未参与 v3.1 的修订起草或复算脚本编写。
- **历史上下文重叠如实声明**：本 Antigravity 会话曾起草前代 v3.0 架构简报，其子代理曾起草 v3.0 暂定 Scientist 草案（`1639b919`）与 v3.0 审查报告（`cb1c3bdf`）。按 `AGENTS.md` §4 与 `agents/reviewer.md`，跨模型族文本复核**不等于**独立实验证据；本报告不对未运行的轨迹、恢复率或耗时作实证背书。
- **被审对象（基于 `HEAD fa4c0982` 上的未提交 v3.1 工作区快照，LF 归一化 SHA256 前 16 位）**：
  1. `reports/pdd_v3_architecture_20261128/revision_ledger_v3_1.md`（`7,331 B`，`4f9406df5e642c91`）
  2. `results/pdd_v3_architecture_20261128/v3_1_static_algebra_check.json`（`160,788 B`，`39733c7426e35ed6`）
  3. `reports/pdd_v3_architecture_20261128/scientist_design.md`（`66,423 B`，`3100f28c769f38e2`）
  4. `designs/pdd_v3_architecture_brief.md`（`59,965 B`，`ab115bbf547681f6`）与镜像 `reports/.../architecture_brief.md`（`60,128 B`，`f93a729575bb50bd`；归一化相对链接后正文 SHA256 严格一致：`b2e302ded6c7637b...`）
  5. `reports/pdd_v3_architecture_20261128/v3_1_scientist_spec.md`（`50,673 B`，`4acfbbc00ad7f1f8`，定向抽查）
  6. `campaigns/pdd_v3/campaign.md`（`10,489 B`，`6859793660a63adb`）、`reports/.../val_trajectory_audit.md`（`10,923 B`，`6c1e357b201c25fb`）、`results/.../scientist_receipt.json`（`6,139 B`，`5a7416ef615a0434`），以及 `docs/current_state.md`、`docs/current_design.md`、`designs/README.md`、`campaigns/README.md`。
- **独立复算与只读纪律**：在共享解释器中以零积分纯静态脚本独立重算了基准 1（$N=8,16,32$）全部拓扑、$s_N$、自然频率、$5N$ 解析 Jacobian 商谱与单模态零空间、`T(0.4 sin t)` 谐波、114 窗离散频响与连续通带下界，以及基准 3（$N=8,16$）三个平衡点、复步 $4N$ Jacobian、`WC Stage-1-Only` 平衡点及 E←E Stage-1 偏置稳定性边界。**严格执行 0 被审文件修改、0 训练更新、0 ODE 轨迹积分、0 新 FOM、0 `private_data/` 或 IEEE-39 `test` 访问。**
- **未检事项声明**：未执行任何轨迹积分、数据生成或模型训练；未通读 `v3_1_scientist_spec.md` 全部逐字段落（仅核对与 `scientist_design.md` 核心公式/台账一致性）；未核验 `v3_1_scientist_spec.md` 签发来源的外部 API 日志（回执已诚实记 `api_request_id: null`）。

---

## 二、总评与最终裁决

- **最终裁决**：**`GO`（有条件放行至代码实现与最小判别 Run 0）**。
- **核心结论**：v3.0 审查提出的 **全部 9 项缺陷（`B1–B3`、`M1–M4`、`m1–m2`）均已实质性闭环（`9/9 CLOSED`）**，所有静态谱、通带下界、平衡点与台账算术经独立复算吻合至机器精度（$<2\times 10^{-15}$），且未发现新的数学定理漏洞、信息泄漏或静态阻断缺陷。
- **放行前提（非重审项，落入 Run 0 冻结配置即可）**：
  1. 维持用户解除 `PAUSED_BY_USER` 后的显式放行授权、`default none`、新 FOM `= 0` 与 IEEE-39 `test` 零访问；
  2. 在 Run 0 冻结配置中补明 1 项合同缺省（**`N2`**：明确基准 1/2 `Stage-1-Only` 同样包含可学习快机械通道 $-b_i T_i(\omega)$ 与冻结 $\hat g$）；
  3. 在 Run 0 基准 3（WC）Stage 1 完成后强制打印 1 项新发现的静态失稳诊断（**`N1`**：监控 $\hat\rho_{EE}=\overline{\hat w_{EE}/w_{EE}^\star}$ 与冻结 $\hat W$ 下的低态 Jacobian 最大实部，临界值 $N=8$ 为 `1.213`、$N=16$ 为 `1.150`）；
  4. Phase 1 扩张仍严格受 Run 0 全部零泄漏、双网格数值资格、含 `PHNN` 的 11 配置 dry-run 及冒烟门禁约束。

---

## 三、v3.0 审查项逐条核销台账（`B1–B3`、`M1–M4`、`m1–m2`）

| 编号 | 核销状态 | 独立复核证据（位置与复算结果） |
|---|---|---|
| **B1** | **`CLOSED`** | `scientist_design.md` §1、§4、§7：Theorem 1 结论 4 明确拆分为 **4a**（必然不可辨的代数端口平移核：$[I,-I]$ 秩 $N$、零维 $N$；WC $[I,I]$ 核 $(v,-v)$）与 **4b**（骨架恢复：诚实承认已知 $M,D$ 且低维驱动可剖除时纯运动弱式仍可恢复，明确 LUPI 差距来源为不可剖除通道、$\langle\phi',M\omega\rangle$ 噪声 $O(\Omega_k)$ 放大及 WC 非线性 sigmoid）；净力分部积分符号 $y_{\mathrm{net}}=\langle\phi',M\omega\rangle-\langle\phi,D\omega\rangle=\langle\phi,\tau_e-\tau_m\rangle$、$+bT(\omega)$ 与摆阵重力 $-\langle\phi,G\sin\theta\rangle$ 全部精确；**`No-Privilege Two-Stage`** 已升为 G3 主对照（基准 1 锚格 3 种子 + 噪声两端各 2 种子 + 基准 3 $N=8$ 3 种子 + Run 0 双系统），`Same-Arch` 降为次级优化对照。 |
| **B2** | **`CLOSED`** | `scientist_design.md` §2：(2a) 误差界显式保留非线性速度泄漏项 $\bar E_{(h'(a)-\ell)\odot\omega}$，$\Omega_{\min}^{-1}\to 0$ 推论严格限定于线性 $h$ 或 $\partial_\omega\psi_a\equiv 0$ 且宽带一致强制，并明示 Mori–Zwanzig 非对角 $\Gamma_\omega$ 边界；(2b) 显式承认完全图零共模反例，A3 要求非零共模转速或大摆角非线性，封闭格在历史前叠加独立共模初速 `Uniform[-0.4, 0.4] rad/s`；(2c) 统一权重 $w=\sqrt{N}/2$、$A_w=[I_N;w\mathbf{1}^\top]$ 同步作用于 $y,X_c,X_g,L$ 与残差，$n=RK(N+1)$，Young 不等式下界 $\min\{h_c g_s^2/(h_c+2m_g^2),h_c/2\}$、SVD 投影 $Q=I-(AL)(AL)^+$、以及共模记忆放大 $C_N=\sqrt{\frac{N}{N+1}}\sqrt{1+N^2/4}$ 与独立噪声放大 $D_N=\sqrt{\frac{N}{N+1}}\sqrt{1+N/4}$ 均验证精确，过度宣称已删。 |
| **B3** | **`CLOSED`** | `scientist_design.md` §4–§5 & 独立复算：(1) 基准 1 三个确定性系统（$N=8,16,32$）自然频率对齐 `[0.800, 1.833/3.307/4.503] Hz`；完整 $5N$ 解析 Jacobian 复算与 JSON 误差 $\le 1.4\times 10^{-15}$（6 个开放格全严格左稳，快慢分离 `8.447×–20.385×`，快振荡 `[0.446, 4.670] Hz`）；$\Delta_e=\cos(\varphi_e-\eta_e)-1$ 消除参考一阶导；`T(0.4 sin t)` 1/3/5 次谐波 `0.37757 / -0.08191 / -0.02069` 精确；114 个 `sin^4` 窗在 `[0.44, 5.00] Hz` 的网格下界 `0.74019657`、Lipschitz `2.41092`、连续下界 `0.73959 > 1/√2`、DC 残差 `<1.5e-9`（修正后 `<1e-15`）均精确吻合；已删除“删 0.5 Hz”分支并将 `Bounded-Recal` 扩至 `[0.50, 2.20] c_hat` 且诚实标注不保证含真值。(2) 基准 3（WC，$N=8,16$，26/56 与 52/240 边）补齐全部 6 项参数与 Dale 符号一致 STP，三平衡点 `low(0.15,0.15)`、`mid(0.1945,0.1904)`、`high(0.8694,0.8468)` 独立复算残差 $<1.4\times 10^{-15}$，低/高态全 $4N$ 谱均严格左稳（$N=8$: `-0.3164 / -0.3000`；$N=16$: `-0.2979 / -0.3000`），低态 E/I 阻尼振荡 `1.097212 Hz` 精确，且**独立复算确认真值下 `WC Stage-1-Only` 严格单稳（仅 `0.15`）**；`WC Stage-1-Only` 与前向 sigmoid 无 logit 的 `WC No-Privilege Two-Stage` 均已精确定义，动态占比降级为 Run 0 待测项。 |
| **M1** | **`CLOSED`** | `scientist_design.md` §2–§3：Theorem 2(1b) 补齐 $\eta=0,h\equiv 0,\partial_a\psi_e\equiv 0$；Theorem 2(2) 改写为任意连通非二分图可实现的单空间模态 $\theta=\theta_\star+uf(t)$（独立复算 $N=8,16,32$ 上 $[A\operatorname{diag}(\alpha),I]$ 秩精确为 `8, 16, 32`、零维 `12, 24, 48`）；Theorem 3 区分具体 $5N$ 全局 Lipschitz 闭合与历史梯度耦合逃逸族 $\dot a=-\nabla_a U_c-\Lambda_a a$，并修正 WC 严格不变域 $[0,(1+\alpha_r)^{-1}]^N\times\mathbb{R}^{3N}$ 及任意有限初态全局解；Theorem 1 结论 3 统一为列归一化非负 KKT + 不可表征条件 + 投影有效刚度 $q_S^{\mathrm{eff}}$，撤销空洞 $\sqrt{s}$ 界与 `[0.2, 1.8] c*` 传递错误。 |
| **M2** | **`CLOSED`** | `scientist_design.md` §6–§7：噪声主轴升级为 $(\sigma_H,\sigma_X,\sigma_P)\in\{(0,0,0),(.01,.01,.01),(.03,.03,.03)\}$ 全轨迹与聚合端口同步加噪、同时间戳共享底样；禁止 $\mathcal{O}_2$ `excitation` 进入 CV/批次分层，改用训练运动可得结局 + 折内历史能量分位并按同源簇整组隔离；公开 `S_e_scale=1.0` 与生成器 `env_gain_true=1.8` 彻底分离；取消硬截断 `init_bound` 改为参考扣减有限线性编码器输出；4 折弱式 CV 明文要求 $Q_T,d_{j,T},\hat b_{T,\alpha}$ 全部折内估计且验证折直接前向预测不重投影；Stage 2 归一化仅用训练折冻结。 |
| **M3** | **`CLOSED`** | `scientist_design.md` §7：`NRI/ACD` 与 `NRI+Port` 改为吃完整训练轨迹并跨事件池化系统级边概率 $q_{ij}$；`PHNN/LGNN` 给出完整 $5N$ 开放端口哈密顿方程及精确功率恒等式 $\dot H$；`SINDy-AE` 首次 STLSQ 剪枝推迟至第 128 步（Phase B 末）并于第 224 步锁支撑，明文限定物理边读出映射；两类无特权基线均保留 $0.05\mathcal{L}_{\mathrm{cons}}$ 并以 $\mathcal{O}_0^{\mathrm{traj}}$ 消除符号混淆。 |
| **M4** | **`CLOSED`** | `scientist_design.md` §8：修正 v3.0 的 `158` vs `128` 汇总笔误，新总账逐行复算精确（`9+30+40+24+12+18+28 = 161` 独立 ODE 拟合，`161×256+128 = 41,344` 主课程更新，毛计 `170`、去重 `9` 严格绑定合同哈希，另 `12` 非 ODE）；`PHNN` 二阶反传纳入 11 配置 dry-run（含 $N=32$ 内存探针与 4 GiB 门禁，6 次正式运行按 `3–8 CPU h` 情景预算，全套 `100–180 h` + `4–6 周` 工程，实测标 `unknown`）；`GraphNeuralODE` 参考读出缓存限于单次 rollout 且不 `detach`。 |
| **m1** | **`CLOSED`** | `val_trajectory_audit.md`、简报与 `v3_1_static_algebra_check.json` L6094–6186：`pilot_0239`（`0.42 s`, 32 帧）已更正为第 7 快，`expansion_0186` 与 `pilot_0152`（均为 `0.33 s`, 23 帧）准确标为 `window_settled`，相对 COI 转速 `< 0.005 pu` 口径清晰。 |
| **m2** | **`CLOSED`** | 镜像措辞已改为“正文一致镜像”且经归一化校验完全一致；`scientist_receipt.json` 已登记 `gpt-6-astra` 于 `2026-11-28 19:50 (UTC+8)` 签发的 `v3_1_scientist_spec.md` 并原样保留历史 v3.0 HTTP 400 回执（诚实记录 `api_request_id: null` 与 root 提供来源）；`campaign.md`、`docs/current_state.md`、`docs/current_design.md` 及两处 `README.md` 均同步维持 `PAUSED_BY_USER / default none`。 |

---

## 四、v3.1 新发现的实质风险与 Run 0 落地注意事项（无新 BLOCKER）

### 1. `[N1 · MAJOR · Run 0 必查机制风险]` 基准 3（Wilson–Cowan）E←E Stage-1 正向偏置存在狭窄的低态失稳悬崖，且 WC 未配 `Bounded-Recal`

- **位置**：`scientist_design.md` §5（L167–205）、§8（L272）。
- **独立静态 Jacobian 复算证据（0 积分）**：
  - 在基准 3 真值中，E←E 强易化调制为 $\beta_{EE}/w_{EE}=+0.80$；因 $r_\star=0.15$ 强非对称且高态 $a=2.68$，高发放跃迁过程中 $a>0$ 会使瞬时耦合 $K_{EE}\in[1.00,1.80]W_{EE}$，导致 Stage 1 弱式估计 $\hat w_{EE}=\rho_{EE} w_{EE}^\star$ 产生正向有效偏移（$\rho_{EE}>1$）。
  - 冻结 $\hat W=\rho_{EE} W_{EE}^\star$ 后对完整 $4N$ 静息态 $r=(0.15,0.15)$ 做静态 Jacobian 扫描：
    - **当 $\rho_{EE}\ge 1.15$（仅高估 `15%`）**：`WC Stage-1-Only`（$\beta=m=0$）即**凭空产生稳定的高发放平衡点**（$r\approx(0.818,0.838),\,\max\operatorname{Re}\lambda=-0.300$），威胁 G1 的双稳态对比；
    - **当 $\rho_{EE}>1.2128$（$N=8$）或 $\rho_{EE}>1.1501$（$N=16$）**：`WC PDD-Frozen`（即便取真实 Stage 2 参数）与 `WC Stage-1-Only` 的**低发放平衡点双双失去线性稳定性**（$N=8$ 下 $\rho_{EE}=1.22, 1.30, 1.40$ 时 $\max\operatorname{Re}\lambda=+0.182, +3.751, +7.016\ \mathrm{s}^{-1}$）；
    - 由于 `PDD-Frozen` 冻结 $\hat w$ 且限定 $\beta_{EE}\in[0,0.8]\hat w$（低态 $a=0$ 时无法向下修正 $K_{EE}$），连把线性负反馈 $G_{ur}$ 放大 $250\times$ 至 `5.0`（$\Lambda_u=5.0$）在 $\rho_{EE}=1.40$ 下仍无法镇定低态（$\max\operatorname{Re}\lambda=+1.027\ \mathrm{s}^{-1}$），且基准 3 未包含 `Bounded-Recal` 消融。
- **不构成 `HOLD` 的理由与 Run 0 处理要求**：实际 $\hat\rho_{EE}$ 取决于带通窗 `[0.5, 5.0] Hz` 内“低态振荡（$a\approx 0$）”与“快上升沿（$a$ 滞后于 $d$）”的能量权重，静态代数无法断言其必超过 `1.15`。因此**放行进入 Run 0**，但须在 Run 0 完成 WC Stage 1 后立即记录 $\hat\rho_{EE}$ 与冻结 $\hat W$ 下的低态谱实部；若实测 $\hat\rho_{EE}\ge 1.15$ 或低态失稳，按 §10.4 形成新冻结小版本（如在 WC 补入允许下调的 `Bounded-Recal` 盒 `[0.50, 1.30] w_hat`，或增大低态稳定裕度）。

### 2. `[N2 · MINOR · 实现前合同补口]` 基准 1/2 `Stage-1-Only` 需在 Run 0 配置中明确包含快机械通道 $-b_i T_i(\omega)$ 与冻结 $\hat g$

- **位置**：`scientist_design.md` §4（L134）、§7（L233）、§10.1（L397）。
- **事实与要求**：v3.1 已明文补齐 `WC Stage-1-Only`（L205），但在为基准 1 新增快机械通道 $-b_i T_i(\omega)$ 与损耗 $\hat g$ 后，未同步重述基准 1/2 `Stage-1-Only` 的完整方程。实现 Run 0 前须在冻结配置中明文锁定：基准 1/2 `Stage-1-Only` 冻结 Stage 1 的 $(\hat S_c,\hat c,\hat S_g,\hat g)$，置 $\beta=a=e=0$ 并删除调制与环境读出，**保留可训练的 $\hat\tau_m=S_p p-b\odot T(\omega)$（含线性 $\dot p$、可学习 $b$ 与 $p_0$ 编码器）**，避免 G1 混入快机械通道错配。

### 3. `[W1–W3 · 非阻塞 Run 0 观察项]`

- **`W1`（基准 1 G3 骨架差距的经验依赖）**：因公开字典 $T_i(\omega)$ 允许 `No-Privilege Two-Stage` 在 $L$ 中直接剖除，基准 1 在零噪下该无特权对照仍保留骨架可辨识性（v3.1 §1 已诚实承认）；`(.01,.01,.01)` 锚格能否达到 G3 `F1 >= +0.15` 将主要取决于 $O(\Omega_k)$ 噪声放大与 $2N$ 干扰列条件数，而端口分离（4a）与基准 3 非线性读出提供最稳固的结构性 LUPI 差距。
- **`W2`（WC 开放记忆幅值与低态阻尼时间尺度）**：WC 高态下静态与调制电流合计 `2.213`，而适应项 `-1.5 m = -0.112`（约 `5%`）、驱动 `u = -0.024`（约 `1%`）；且低态 `1.097 Hz` E/I 模态衰减时间为 `0.207 s`（1 s 衰减 99.2%），Run 0 须实测 `[1,3] s` 与 `[2,4] s` 窗的有效激励及 `m` 对长时轨迹的真实贡献（v3.1 §5 已诚实将原 RMS 目标撤为待测）。
- **`W3`（实现变更与来源备忘）**：取消编码器 `init_bound` 硬截断后需改动现有 `OSGKDGraph._GraphEncoder` 并重测参数量（原 `1,928` 参）；另记 `scientist_receipt.json` 中 v3.1 `gpt-6-astra` 签批的 `api_request_id` 为 `null`（由 root 交付文本归档，非本机 CLI 直连回执），审计链已透明记录。
