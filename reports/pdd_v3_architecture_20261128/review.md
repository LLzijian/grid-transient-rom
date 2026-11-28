# PDD v3 架构简报与暂定 Scientist 有界设计独立只读审查报告

---

## 一、审查身份、范围、输入校验与未检事项声明

- **审查者实际宿主身份**：本报告由 **Antigravity 只读 `research` 子代理**（会话 ID `cb1c3bdf-e6f2-49a4-8dcd-a272b8507693`）执行。本子代理未参与本轮 PDD v3 架构简报、`val` 核查或暂定 Scientist 有界设计的起草，且与起草者上下文完全隔离。
- **利益 / 模型族重叠如实声明**：本子代理与父会话 `root`（`eb5e0ef2-885f-48d4-b880-5f99e906f32f`，Gemini 4 Neon）及暂定 Scientist 设计起草子代理（`1639b919-2a4c-4032-b93d-7768bcefc975`，`gemini-4-neon-xhigh`）**同属 Antigravity 宿主 / 模型族生态**。按 `AGENTS.md` §4 与 `agents/reviewer.md`，上下文隔离**不等于**跨厂商模型独立性（尚未完成 `gpt-6-astra` 真实审查），亦**不能替代**真实实验证据。
- **冻结审查清单校验**：已读取并校验 `results/pdd_v3_architecture_20261128/review_inputs.json`：
  - 清单文件自身 SHA256（原始与 LF 归一化一致）：`c750274d68c74b683f2fdcd06d8b02054d0561b101da13ac5be06b11bc60e729`（与任务指令完全吻合）。
  - 清单内全部 8 份冻结文件均已逐一校验磁盘字节数与 LF 归一化 SHA256，**8/8 全部 `match_lf = True`、`bytes_disk == bytes_manifest`**：
    1. `designs/pdd_v3_architecture_brief.md`（`34,680 B`，`fe497fe8...`）
    2. `campaigns/pdd_v3/campaign.md`（`7,225 B`，`564d7129...`）
    3. `reports/pdd_v3_architecture_20261128/architecture_brief.md`（`34,922 B`，`62aa7afa...`）
    4. `reports/pdd_v3_architecture_20261128/positioning_assessment.md`（`15,428 B`，`f7a48424...`）
    5. `reports/pdd_v3_architecture_20261128/val_trajectory_audit.md`（`10,199 B`，`0f1a1b3c...`）
    6. `reports/pdd_v3_architecture_20261128/scientist_design.md`（`74,059 B`，`ab98674c...`）
    7. `results/pdd_v3_architecture_20261128/val_audit_summary.json`（`46,752 B`，`4aab253f...`）
    8. `results/pdd_v3_architecture_20261128/scientist_receipt.json`（`3,553 B`，`e94165f7...`）
- **对照核查范围**：额外只读核查了 `AGENTS.md`、`kiki.md`、`agents/reviewer.md`、`agents/scientist.md`、`reports/osgkd_v2_wp2_layer1_20261127/resource_budget.md`、`paper/osgkd_v2_methods_and_theory.md`、`src/kiki/models/osgkd.py`、`src/kiki/models/osgkd_graph.py`、`src/kiki/models/osgkd_baselines.py`、`src/kiki/data/osgkd_synthetic.py`、`experiments/osgkd_v2_wp2_layer1_20261127/models.py`、`experiments/osgkd_v2_wp2_layer1_20261127/negative_controls.py`、`experiments/osgkd_v2_layer1/escape_example.py`。除静态代数/特征值/不动点公式验算外，**严格执行 0 文件修改、0 训练更新、0 ODE 积分、0 `private_data/` 与 IEEE-39 `test` 访问**。
- **未检事项声明**：
  1. 未执行任何 ODE 数值仿真、生成器运行或训练脚本（受只读红线约束），因此基准 1/3 的动态轨迹指标（如调制 RMS 占比、实际失稳比例）仅通过静态特征值与平衡点代数核验，未做动态实测；
  2. 未逐文件重算 `artifacts/` 中原始 `rows.json`/`metrics.json`/`spectrum.json` 的哈希与记录（依赖 `val_audit_summary.json` 汇总层与跨表一致性校验）；
  3. 未通读 `pendulum.py` 与 `osgkd_fitting.py` 全文；
  4. 跨厂商模型复核（`gpt-6-astra`）因本机 `codex-cli 0.144.4` 限制尚未完成。

---

## 二、总评与放行建议

- **总评结论**：**方向正确，但当前冻结版本存在 3 项实质性 `BLOCKER` 缺陷，不建议按现状直接放行进入代码实现或 Run 0。**
  - **肯定之处**：从 OSGKD v2 端到端五合一收束为“LUPI 信息不对称 + 两阶段解耦 + 3 条可辨识与适定性定理 + 大样本失稳评测”是面向 NeurIPS 2027 的正确战略转向；治理红线（新 FOM = 0、IEEE-39 `test` 零访问、`gpt-6-astra` 真实 HTTP 400 回执无冒充、`DRAFT / default none`）执行严格；Theorem 1 测试函数闭式范数恒等式（$\|\Phi_{m,q}\|_{L^2}$）、Theorem 3 奇对称逃逸上界（$1.2566$）与 Wilson–Cowan $[0,1]^N$ 前向不变性均经核验精确成立；单进程 CPU float64 算力总体完全落在 NeurIPS 2027 5 个月周期内。
  - **致命风险**：若不先修补下列 **3 项 `BLOCKER`** 就开工，项目将同时面临：（1）**论文头条（LUPI）被审稿人一剑封喉**（在当前基准 1/2 已知 $M, D$ 且 $\dot p \in \mathcal{F}_{\mathrm{env}}$ 下，纯运动 $M\dot\omega+D\omega$ 跑 Stage 1 即可恢复同精度静态骨架，直接推翻 Theorem 1 结论 4 并击穿 G3）；（2）**Theorem 1 证明存在三处硬伤**（$\tau_{\mathrm{env,cm}}(a)$ 非线性速度项破坏 $\Omega_{\min}^{-1}\to 0$ 收敛；零共模完全图精确反例使 $\sin\theta_{\min}=0$；增广行权重 $\sqrt{N/4}$ vs $\sqrt{1/4}$ 冲突且放大残差/噪声 $O(N)$）；（3）**Run 0 极高概率在物理参数层直接返工**（基准 1 全部自然频率 `0.11–0.58 Hz` 落于 Stage 1 `1–3 Hz` 主通带之下，且 $\dot a$ 负偏置击穿 `PDD-Frozen` 与 `[0.75, 1.40]`；基准 3 Wilson–Cowan 标称参数甚至不存在高发放双稳态平衡点且强过阻尼）。
- **放行建议**：**`HOLD / REVISE_BEFORE_GO`（退回修订后再放行）**。在补齐 **B1–B3（必改）** 与 **M1–M4（强烈建议同步修完）** 并就 `gpt-6-astra` 暂定绑定获得用户显式书面裁决后，方可进入代码实现与 Run 0。

---

## 三、逐条关键缺陷清单

### BLOCKER（放行前必须修正）

#### [B1] LUPI 头条主张与 G3 门禁的结构性漏洞：基准 1/2 方程允许纯运动观测 $\mathcal{O}_0^{\mathrm{traj}}$ 直接运行 Stage 1 恢复静态骨架，且 `No-Privilege Same-Arch` 被强令跳过 Stage 1 造成对照混淆

- **位置**：`designs/pdd_v3_architecture_brief.md` L106（Theorem 1 结论 4）；`reports/pdd_v3_architecture_20261128/scientist_design.md` L97（结论 4）、L287（基准 1 $\dot p$）、L311（表 4.4 `No-Privilege Same-Arch`）、L330（G3 门禁）。
- **数学与技术证据**：
  1. 在基准 1/2 中，$M, D \in \mathcal{C}$ 为冻结公开常数（L49），且主动驱动 $\dot p_i = -\lambda_{p,i}p_i - \gamma_{p\omega,i}\omega_i - \gamma_{pt,i}\tanh(p_i)$（L287，$\lambda_p \in [0.40, 0.65]\ \mathrm{s}^{-1} \ll \Omega_{\min}$）**本身就属于严格真节点局部记忆类 $\mathcal{F}_{\mathrm{env}}$**。
  2. 仅用训练期纯运动轨迹 $(\theta(t), \omega(t))$（**完全不用 $\tau_m, \tau_e \in \mathcal{O}_1$**），对 $M\dot\omega + D\omega = \tau_m - \tau_e$ 施加 $C_c^2(I_k)$ 零均值带通测试函数 $\phi_k = \Psi'_k$（如 $\sin^3(\pi u/L)$ 包络，使 $\Psi,\Psi',\Psi''$ 在端点均为零），无需任何数值求导即可得：
     $$y_{r,k}^{\mathrm{net}} := \langle\phi'_k, M\omega^{(r)}\rangle - \langle\phi_k, D\omega^{(r)}\rangle = \langle\phi_k, F_c + F_{\mathrm{sym}}\rangle + \langle\phi_k, S_e e - F S_p p + \tau_{\mathrm{env,cm}}(a)\rangle.$$
     对 $\langle\phi_k, p_i\rangle$ 分部积分后，其速度耦合项 $\gamma_{p\omega,i}\langle\Phi_k, \omega_i\rangle$ 落入**同一个**干扰空间 $\operatorname{col}(\Phi_{\mathrm{loc}})$ 被 $Q_\perp$ 精确消去，剩余项同样按 $O(\Omega_{\min}^{-1})$ 衰减！
  3. **对结论与实验的破坏性影响**：
     - **推翻 Theorem 1 结论 4**：在当前基准 1/2 方程（已知 $M, D$ + 严格真节点局部 $\dot p$）下，取消 $\mathcal{O}_1$ **并不导致**静态骨架 $c^\star$ 失去唯一可辨识性！无 $\mathcal{O}_1$ 真正恒不可辨的是**端口分离 $(\tau_m, \tau_e)$ 与 $(p,e)$ 归属**（Theorem 2 情形 4 的 $N$ 维平移规范）；静态骨架仅在 $\tau_m \notin \mathcal{F}_{\mathrm{env}}$（含宽带/瞬时或非局部驱动）、未知 $(M,D)$ 尺度、或非线性读出系统（如 Wilson–Cowan $S(\cdot)$）叠加观测噪声时，才因缺少 $\mathcal{O}_1$ 丧失可辨识性或稳健性。
     - **击穿 G3 公平性**：表 4.4（L311）强制 `No-Privilege Same-Arch` **跳过 Stage 1** 直接端到端拟合，把“两阶段解耦增益”与“特权信息 $\mathcal{O}_1$ 增益”混在一起。任何严苛的 NeurIPS 审稿人都会要求补测 **`No-Privilege Two-Stage`**（用 $M\dot\omega+D\omega$ 跑 Stage 1 + 无端口 Stage 2）——在当前无噪训练 `[0,4] s` 轨迹上，后者同样可拿到静态边 $\mathrm{F1}\approx 1.0$，使 G3 的 `F1 提升 >= +0.15` 彻底失效。
- **最小必要修正方案**：
  1. **精确收紧 Theorem 1 结论 4 与论文 LUPI 定理边界**：明确区分两层结果——① 无 $\mathcal{O}_1$ 时，端口分离 $(\tau_m, \tau_{\mathrm{env}})$ 与 $(p, e)$ 归属存在精确 $N$ 维不可辨识规范（Theorem 2 情形 4，即使已知 $M,D$ 且无噪也无法消除）；② 静态骨架 $c^\star$ 在无 $\mathcal{O}_1$ 下失去可辨识性或稳健性的精确条件为：$\tau_m \notin \mathcal{F}_{\mathrm{env}}$（如含快变/瞬时/非局部机械或控制驱动），或训练运动轨迹含观测噪声（二次分部积分 $\langle\phi'_k, M\omega\rangle$ 将 $\omega$ 噪声放大 $O(\Omega_k)$ 倍，而基准 3 非线性 $S^{-1}(\tau_r\dot r + r)$ 在噪声下无法弱式线性化甚至产生 `NaN`）。
  2. **基准 1 与 G3 同步升级**：在基准 1 锚格（至少单种子充分性，建议 3 种子）新增 **`No-Privilege Two-Stage`** 对照；同时在基准 1 中① 对训练轨迹施加与部署同口径的观测噪声（见 M2），和/或② 为 $\tau_m$ 引入符合电网/机器人实际的宽带或瞬时局部/非局部驱动成分（使 $\tau_m \notin \mathcal{F}_{\mathrm{env}}$），从而让 $\mathcal{O}_1$ 相对 `No-Privilege Two-Stage` 的增益同时在端口分离、含噪/宽带驱动骨架恢复以及基准 3 非线性读出上真实可检验。相应更新 G3 门禁口径。

---

#### [B2] Theorem 1 存在三处实质性数学漏洞：$\tau_{\mathrm{env,cm}}(a)$ 非线性速度泄漏破坏 $\Omega_{\min}^{-1}\to 0$；零共模完全图反例推翻 Step 1 的 $\sin\theta_{\min}>0$ 证明；增广行权重前后矛盾且放大残差/噪声 $O(N)$

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` §1.2（L60–63）、§1.3 结论 1–2 与 Step 1–2（L82–90, L102–110）、§3.1（L218–219）。
- **数学证据**（逐项验证）：
  1. **漏洞 (2a)：$\tau_{\mathrm{env,cm}}(a)$ 的 $O(1)$ 非线性速度泄漏与 L90 错误收敛宣称**
     - 按 L60、L62、L284，$\tau_{\mathrm{env,cm},i}(a) = \alpha_{1,i}\tanh(a_i) + \alpha_{2,i}\tanh^2(a_i)$，且 $\dot a_i$ 含 $\gamma_{a\omega,i}\omega_i$（$\gamma_{a\omega}\in[0.45,0.70]$）。分部积分得：
       $$\langle\phi_k, \tau_{\mathrm{env,cm},i}(a)\rangle = -\gamma_{a\omega,i}\langle\Phi_k, \tau'_{\mathrm{env,cm},i}(a_i)\omega_i\rangle + \langle\Phi_k, \tau'_{\mathrm{env,cm},i}(a_i)(\lambda_{a,i}a_i - \tilde\psi_{a,i})\rangle.$$
       因 $\tau'_{\mathrm{env,cm},i}(a_i)$ 随未知 $a_i(t)$ 时变，首项**不在** $\operatorname{col}(\Phi_{\mathrm{loc}})$ 内，不能被 $Q_\perp$ 投影消去。
     - L90 宣称“当 $\Omega_{\min}\to\infty$ 且无噪时，估计误差按 $\Omega_{\min}^{-1}$ 严格收敛至零”，这在 (A1)–(A3) 下**双侧不成立**：
       - 若**固定轨线**令 $\Omega_{\min}\to\infty$：由 $\langle\phi_k,\sin\varphi_e\rangle = -\langle\Phi_k,\cos\varphi_e\cdot B^\top\omega\rangle = O(\Omega_{\min}^{-1})$，静态最小奇异值 $\gamma_{\mathrm{stat}}(\Omega_{\min})$ 至少以 $O(\Omega_{\min}^{-1})$（光滑轨线实为超多项式）衰减，L89 右端分母与分子同步/更快趋零，估计误差**不收敛于 0**；
       - 若**随 $\Omega_{\min}$ 提高激励带宽**以保持 $\gamma_{\mathrm{stat}} = \Theta(1)$：则在 $\Omega_{\min}$ 处须有 $\Theta(1)$ 角振荡，必然推出速度能量 $\bar E_\omega = \Theta(\Omega_{\min})$，从而 $\bar E_{\psi_a} = \Theta(\gamma_{a\omega}\Omega_{\min})$，代入 L87 得 $\Omega_{\min}^{-1} L_{\mathrm{cm}}\bar E_{\psi_a} = \Theta(L_{\mathrm{cm}}\gamma_{a\omega}) = O(1) \not\to 0$！L87 将其写作 $O((T_{\min}^{-1}+\Lambda_{a,\max}+1)/\Omega_{\min})$ 隐含假设了 $\bar E_{\psi_a}=O(1)$，与 $\psi_a$ 依赖 $\omega$ 矛盾。
     - （另注：若推广至真实约化系统如 IEEE-39，$\dot e$ 的 Mori–Zwanzig 记忆核一般含**非对角**速度耦合 $\Gamma_{\omega,ij}\omega_j$，在小角区将与 $L_c\theta$ 精确共线，因此 $\Gamma_\omega$ 对角前提在适用边界中须重点标示。）
  2. **漏洞 (2b)：Step 1 线性无关证明（L103）在零共模转速下被精确反例推翻**
     - L103 称“小角线性化极限下，$L_c+\operatorname{diag}(\tilde\gamma)=0$ 的非对角元 $-c_{ij}=0$ 立推 $c=0$，故对所有 $N\ge 2$ 必有 $\sin\theta_{\min}>0$”。
     - **精确反例**：在完全候选图 $E=K_N$（L49）上，只要轨迹共模转速为零（$\mathbf{1}^\top\omega(t)\equiv 0$，如封闭系统或中心化角），$(L_c+\operatorname{diag}(\tilde\gamma))\theta=0$ 仅在子空间 $\mathbf{1}^\perp$ 上成立，等价于 $L_c+\operatorname{diag}(\tilde\gamma) = v\mathbf{1}^\top$。取完全图等权 $c = c_0\mathbf{1}_m$ 与 $\tilde\gamma = -N c_0 \mathbf{1}_N$，有 $L_c + \operatorname{diag}(\tilde\gamma) = c_0(N I_N - \mathbf{1}\mathbf{1}^\top) - N c_0 I_N = -c_0\mathbf{1}\mathbf{1}^\top$，在 $\mathbf{1}^\perp$ 上**恒等于零**！即 $\Phi_c \mathbf{1}_m = \Phi_{\mathrm{loc}}(N\mathbf{1}_N)$，**对任意 $N\ge 2$ 在小角零共模极限下精确有 $\sin\theta_{\min} = 0$**。
  3. **漏洞 (2c)：全局求和增广行权重矛盾、遗漏 $\Phi_{\mathrm{loc}}$ 增广、且破坏结论 2 的无 $N$ 误差界**
     - **自相矛盾**：Theorem 1 L83 规定标量增广行权重为 $w = \sqrt{N/4}$，而 §3.1 算法 L218 却写成 $w = \sqrt{1/4}$（相差 $\sqrt{N}$ 倍！若按 L218 取 $w=1/2$，除以 $K_{\mathrm{tot}}=RMN$ 后余弦块系数为 $\frac{1}{N}+\frac{4}{N^2}$，仍按 $1/N$ 退化）；
     - **实现漏洞**：L219 未要求对 $\Phi_{\mathrm{loc}}$ 同步追加增广行 $w\mathbf{1}^\top\Phi_{\mathrm{loc},(r,k)}$，若遗漏则 $Q_\perp$ 无法在增广行中消去速度记忆项 $w\sum_i S_{e,i}\gamma_{e\omega,i}\langle\Phi_k,\omega_i\rangle$；
     - **定理失效**：即便统一采用 $w=\sqrt{N/4}$（并修正行数归一化 $K_{\mathrm{aug}}=RM(N+1)$ 带来的 $\frac{N}{N+1}$ 因子），对标量全局和行乘以 $\sqrt{N/4}$ 会同步把增广行中的**同号共模记忆残差**的平方贡献放大为 $\frac{1}{N}(w\mathbf{1}^\top r)^2 = \frac{N^2}{4}\bar r^2$（$L_2$ 范数放大 $O(N)$），并把 i.i.d. 噪声方差放大 $N/4$ 倍！因此用 $\gamma_{\mathrm{stat}}^W$ 替代 $\gamma_{\mathrm{stat}}$ 虽消去分母的 $1/N$，却同步把分子放大 $O(\sqrt{N})\text{–}O(N)$，L87/L89 的无 $N$ 分子上界对增广系统**不再成立**。（这是物理信噪比极限：单条边 $g_e$ 对全局和仅贡献 $2\Delta\cos\varphi_e=O(1)$，而 $N$ 个节点的共模环境项累加为 $O(N)$，重加权一行不能凭空消除该缩放。）
- **最小必要修正方案**：
  1. 对 (2a)：若保留精确 $\Omega_{\min}^{-1}\to 0$ 渐近陈述，需在 (A2) 要求 $\tau_{\mathrm{env,cm}}(a)$ 线性（可并入 $Q_\perp$）或 $\partial_\omega\psi_a\equiv 0$，并在 (A3) 显式以“一致强制性宽带轨线族（$\gamma_{\mathrm{stat}}\sin\theta_{\min}\ge c_0>0$ 且 $\bar E_{\tilde\psi_e},\bar E_{\psi_a}=O(1)$）”表述；否则应取消 $\to 0$ 宣称，诚实保留有限带宽下不可消去的非线性速度泄漏地板 $O(L_{\mathrm{cm,nl}}\gamma_{a\omega}\bar E_\omega/\Omega_{\min})$。
  2. 对 (2b)：在 (A3) 与 Step 1 中明示需要**非零共模转速激励**（或大摆角非线性分离）以保证 $\sin\theta_{\min}>0$，并在封闭零共模格点（`static_closed` / `strong_mod_closed`）核查条件数。
  3. 对 (2c)：统一 L83 与 L218 权重及 $K_{\mathrm{aug}}$ 归一化、同步增广 $\Phi_{\mathrm{loc}}$，并在结论 2 中如实写出增广后的分子缩放因子（噪声 $\sqrt{1+N/4}$、共模记忆 $\sqrt{1+N^2/4}$），删去“完全消除 $N$ 退化”的过度表述。（另确认：L106–107 的 $\|\Phi_{m,q}\|_{L^2}$ 闭式恒等式与 L82 未加权 $\gamma_{\mathrm{stat}}$ 经核验精确成立，保留。）

---

#### [B3] 基准 1 与基准 3 的核心物理参数存在实质性设计失效风险：基准 1 自然频率低于 Stage 1 主通带且非对称 $a(t)$ 偏置击穿 `PDD-Frozen` 与 `[0.75, 1.40]`；基准 3 Wilson–Cowan 标称参数缺乏高发放双稳态且强过阻尼

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` §1.1（L37）、§1.3 结论 3（L93–95）、§2.2（L175–195）、§3.1（L205–207）、§3.2（L244–245）、§4.2（L282–288）。
- **静态代数与物理实算证据（0 次 ODE 积分，仅特征值与平衡点核验）**：
  1. **基准 1 物理频带与 Stage 1 测试通带严重错位**：
     按 §4.2 参数（$M_i\in[0.7,1.4]$，$c_e\in[1.2,2.4]$，随机环+$N/2$ 弦）实算 $M^{-1}L_c$ 非零特征根对应的线性自然频率 $f_{\mathrm{nat}}=\sqrt{\lambda}/(2\pi)$：
     - $N=8$：$[0.185,\; 0.550]\ \mathrm{Hz}$（$\Omega_{\mathrm{nat}}\in[1.16, 3.46]\ \mathrm{rad/s}$）；
     - $N=16$：$[0.182,\; 0.540]\ \mathrm{Hz}$；
     - $N=32$：$[0.112,\; 0.582]\ \mathrm{Hz}$。
     然而 §3.1 将 **42/48（87.5%）** 测试函数设在 **`1.0, 2.0, 3.0 Hz`**，仅 6 个设在 `0.5 Hz`！物理信号主频带几乎完全落在主通带之下（根源在于 $T_e^{-1}\le 0.35\ \mathrm{s}^{-1}$ 对应截止频率仅 $0.056\ \mathrm{Hz}$，设计时混淆了角频率 $\mathrm{rad/s}$ 与频率 $\mathrm{Hz}$ 的 $2\pi$ 因子）。这会导致 $\tilde\Phi_{\mathrm{stat}}$ 条件数恶化、加噪后 Stage 1 性能大幅下降；而 Run 0 分支 B（L376）甚至预拟“删掉 `0.5 Hz` 仅留 `1–3 Hz`”，会切断唯一与自然频带重叠的频段！
  2. **基准 1 与基准 3 均系统性违背奇对称核“零偏差”前提并击穿 `PDD-Frozen` 与 `[0.75, 1.40]`**：
     - **基准 1（L284）**：$\dot a_i$ 的电压跌落驱动项 $\gamma_{a,\cos,i}\sum_{e\in S^\star(i)}(\cos\varphi_e - \cos\eta_e)$ 在大摆角与失稳轨迹上**几乎处处为负**（因 $|\eta_e|$ 较小，$\cos\eta_e\approx 0.96\text{–}1.00$）。对平均度 3 节点，摆幅 RMS $\sim 1.0\ \mathrm{rad}$ 时稳态直流偏移 $\bar a_i \approx -1.07 \Rightarrow \bar s_e \approx -2.1 \Rightarrow \mu_e \in [-0.85, -0.50]$；且回归实际估计的是按 $\sin^2\varphi_e$ 加权的投影均值 $\mu_e^{\mathrm{eff}}$，因大摆角时刻 $|\sin\varphi_e|$ 与负 $a_i$ 同步达峰，负偏更剧。结果 Stage 1 估计 $\hat c_e \approx (1+0.75\mu_e^{\mathrm{eff}})c_e^\star \approx 0.45\text{–}0.63\,c_e^\star$（系统性低估 37%–55%）。
       - 对 **`PDD-Frozen`**：冻结 $\hat c_e$ 后，即便拉满 $\beta_e = 0.8\hat c_e$，最大刚度 $1.8\hat c_e$ 仍可能低于真实静刚度 $c_e^\star$，迫使 Stage 2 给 $a(t)$ 注入永久正直流偏置——**重演了 §1.1 用来淘汰旧 $\exp(\tanh s)-1$ 核的同一跨模块直流补偿问题**，并可能使 `PDD-Frozen` 在 G2 中长时 MSE 劣于端到端 v2！
       - 对 **`Stage-2 Bounded-Recal`**：投影盒 $[0.75\hat c_e, 1.40\hat c_e]$ 仅在拍脑袋假设 $|\mu_e|\le 0.35$（L244）下成立，当真值 $c_e^\star \in [1.6\hat c_e, 2.2\hat c_e]$ 时**根本无法包含真值**。
     - **基准 3（L183）**：静息点 $r_\star\approx 0.16$ 距下界 $0$ 仅 $0.16$、距上界 $0.87$ 达 $0.71$（$4.4\times$ 不对称），$\dot a_i \ni \gamma_{ar,i}(r_i-r_{\star,i})$ 导致 $a_i$ 强烈**正偏**（$\mu_{ij}\gg 0$），Stage 1 系统**高估** $w_{ij}^\star$ 40%–65%，$w_{ij}^\star \approx 0.60\,\hat w_{ij}$ 同样跌出 `[0.75, 1.40]`。
  3. **基准 3（Wilson–Cowan）当前参数缺乏高发放双稳态、STP 方向自相矛盾、过阻尼无振荡且缺 6 项参数**：
     - 静态不动点与 2 群体 E–I 雅可比实算：在 §2.2 中点参数（$W_E=1.9, W_I=1.3, \alpha_r=0.15, \gamma_s=5.0, x_0=0.5, r_\star=0.16 \Rightarrow I_0=0.1742$）下，即使设 $m=0, u=0$ 且把 $2/3$ 边 STP 易化拉满（$1.48\times$），在 $\bar r_E > 0.65$ 区域仍**不存在平衡点**（处处 $f_E < 0$，最小残差 $0.0253$，唯一稳态为 $0.16$）；一旦接入慢适应 $-S_m m$（$S_m\in[1.2,1.9], T_m\in[2.5,5.0]$）和负反馈 $-\gamma_{ur}(r-r_\star)$，高发放态立即被强负电流扑灭（$g(0.65)=-0.64$）。核心物理矛盾在于 L180 对兴奋性与抑制性突触施加了**同向**易化 $\beta_{ij}\ge 0$，高活动时同步放大抑制电流，直接破坏了 L162 宣称的“STP 增强 $W_{EE}$ 形成双稳态癫痫”机理！
     - 2 群体线性雅可比在 $r_\star$ 处特征根为纯负实根（`-5.42, -8.91 s⁻¹`；即使极限角 `W_EE=2.1, W_EI=-1.1` 仍为 `-0.36, -13.97 s⁻¹`），系统**强过阻尼、零振荡**，稳定轨线在 `0.3–0.5 s` 内即衰减平直，令 `[1,4] s` 的 5 个 Stage 1 带通窗几乎空转，危及 (A3) 强制性。
     - L182–185 遗漏了 $\gamma_{ur}, \gamma_{ar}, \gamma_{a,\mathrm{sat}}, \gamma_{mr}, \gamma_{ma}, \alpha_{\mathrm{cm}}$ 共 6 个生成器参数的数值范围，且全文未给出基准 3 上 `Stage-1-Only` 的正式定义。
- **最小必要修正方案**：
  1. **基准 1 频带对齐**：调小 $M_i$（如 $[0.25, 0.45]$）或调大 $c_e$（如 $[3.5, 6.5]$）使 $f_{\mathrm{nat}}\in[0.45, 1.35]\ \mathrm{Hz}$（RK4 $\Delta t=0.01\ \mathrm{s}$ 仍远在稳定域内），或将 Stage 1 窗频下延至 $[0.25, 1.5]\ \mathrm{Hz}$ 并同步下调 $T_e^{-1}\le 0.15\ \mathrm{s}^{-1}$，确保自然频带落入主通带且与记忆截止分离 $\ge 8\times$。
  2. **正视非对称偏置与放宽投影盒**：在 Theorem 1 结论 3 将 $\mu_e$ 定义为 $\Phi_{c,e}$ 投影有效均值 $\mu_e^{\mathrm{eff}}$，放弃强失稳下“必然零偏”的过强断言；将 `Stage-2 Bounded-Recal` 盒放宽至有理论依据的区间（如 $[0.55\hat c_e, 2.20\hat c_e]$，覆盖 $|\mu_e^{\mathrm{eff}}|\le 0.7$），在 Run 0 直接并列检验 `PDD-Frozen` 与 `Bounded-Recal` 并实测 $\mu_e^{\mathrm{eff}}$ 分布。
  3. **基准 3 参数定标与补全**：放行前经静态不动点与雅可比特征根定标四分块权重 $(W_{EE}, W_{IE}, W_{EI}, W_{II})$ 与全部 6 个缺失系数——将 E→E 设足够强、对 E 突触用易化而对 I 突触用抑制或弱调制（符神经生理 STP），使 E–I 回路在 $r_\star$ 附近具 $1\text{–}2\ \mathrm{Hz}$ 阻尼振荡（复特征根），明确界定发作机制是“持久双稳态”还是“慢适应终止型长时程癫痫发作（$\ge 0.20\ \mathrm{s}$）”，并明文补全基准 3 的 `Stage-1-Only` 定义（冻结 $(\hat S,\hat w)$，关闭 $\beta, a, m$，仅训线性 $\dot u$ 与 $u_0$ 编码器）。

---

### MAJOR（进入 Run 0 前强烈建议修正）

#### [M1] Theorem 1 结论 3、Theorem 2 情形 1(b)/2 与 Theorem 3 逃逸反例各自存在遗漏前提或与算法方程脱节

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` L93–95, L122–123, L134, L139–144, L223；`designs/pdd_v3_architecture_brief.md` L111–112, L118–119。
- **技术证据**：
  1. **Theorem 2 情形 1(b) 遗漏前提**：在 PDD v3 完整方程中，当 $a(0)\ne 0$ 时，$\tau_{\mathrm{env,cm}}(a(0))\ne 0$ 与 $\gamma_{ea}\tanh(a(0))\ne 0$ 会立即驱动 $\dot\omega(0)\ne 0, \dot e(0)\ne 0$，故“纯幅值轨线 $\theta\equiv\theta_\star, \omega\equiv p\equiv e\equiv 0$”在完整 PDD v3 ODE 上**不存在**！必须补充前提 $\tau_{\mathrm{env,cm}}\equiv 0$ 且 $\partial_a\psi_e\equiv 0$。
  2. **Theorem 2 情形 2 非二分图拓扑不可行性**：原陈述“所有活动边共享同一非零时间波形 $\sin\varphi_e(t)=f(t)$”在含奇圈（如 3-圈）的非二分图上违反 $\sum_{\text{cycle}}\varphi_e=0$，拓扑上不可实现（旧 `negative_controls.py` L109 诚实限定为线性观测伴随模型，而本稿删去了该限定）。
  3. **Theorem 3 逃逸反例与 Stage 2 ODE 脱节**：标量约化 $\dot z = \frac{\pi}{2}K'(z)-z$ 源自梯度耦合储能族 $\dot a = -\nabla_a U_c - \Lambda_a a$（历史 GFTD 储能族），而 Stage 2 的 $\dot a$（L238）**不含 $K'(a)$ 项**！此外，简报 L118 注明“不推出全局 Lipschitz”，而设计稿 L134 称“甚至全局 Lipschitz”（对无状态乘积的具体闭合确实全局 Lipschitz，但两稿自相矛盾）；Wilson–Cowan 适定性（L139）应写明“对 $r(0)\in[0,1]^N$，严格不变域为 $[0, (1+\alpha_r)^{-1}]^N\times\mathbb{R}^{3N}$ 且在该域上全局 Lipschitz 并最终一致有界”。
  4. **Theorem 1 结论 3 定理与算法惩罚项失配及 $\sqrt{s}$ 界过松**：定理 L93 对未加权 $\lambda_1\mathbf{1}^\top\vartheta$ 证界，算法 L223 求解列加权 $\lambda_1\sum_j d_j\vartheta_j$；且未将 $\mu_e$ 定义为投影有效均值时，强调制残差 $r_{\mathrm{mod}}$ 与同列 $\Phi_{c,S^\star}$ 强相关，使 $2\lambda_1\sqrt{s}/\gamma_0^2 \gtrsim 0.6\sqrt{s}\,c_{\max}^\star \gg \frac{1}{2}c_{\min}^{\mathrm{eff}}$，L95 的精确支撑充分条件在 $s=12\text{–}48$ 时空洞不满足。
- **最小必要修正方案**：
  - Theorem 2 情形 1(b) 补齐 $\tau_{\mathrm{env,cm}}\equiv 0, \partial_a\psi_e\equiv 0$；情形 2 改写为小角单空间模态 $\theta(t)=\theta_\star+u f(t)$ 下的 $y(t)=(A\operatorname{diag}(\alpha)c+q)f(t)$（$\alpha_e=\cos\eta_e(u_i-u_j)$，对任意连通图物理可实现，秩仍精确为 $N$、零空间维数仍为 $m$）；
  - Theorem 3 Part 2 明确标注该反例适用于能量/梯度耦合幅值闭合族 $\dot a = -\nabla_a U_c - \Lambda_a a$，并统一简报与设计稿关于全局 Lipschitz 与 Wilson–Cowan 不变域的表述；
  - Theorem 1 结论 3 统一采用列归一化坐标 $\tilde\vartheta_j = d_j\vartheta_j$，改用 $\mu_e^{\mathrm{eff}}$ 定义有效刚度或引入不可表征条件（irrepresentable condition）替代空洞的 $\sqrt{s}$ 界。

---

#### [M2] 隐式信息流、符号冲突与噪声协议漏洞（4 处封口项）

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` L49, L222, L225, L257–258, L262, L285, L296。
- **技术证据**：
  1. **$\mathcal{O}_2$ 激励标签泄漏进训练分层**：L225（Stage 1 4 折 CV）与 L262（Stage 2 批次配平）均按“**激励**/失稳类别分层”。但 `excitation`（如 `latent_overload` vs `boundary_near_separatrix`）属 $\mathcal{O}_2$ 隐藏故障/事件身份（`kiki.md` L24 明文禁用），严禁进入任何训练期分层逻辑。
  2. **噪声横轴（`{0.00, 0.01, 0.03}`）作用范围不清**：旧 `osgkd_synthetic.py` L153–168 仅对 `history`（11 帧）加噪而 `[0,4] s` `targets` 零噪。若直接沿用，Stage 1 在全部噪声档均见零噪数据，Theorem 1 的有限噪声界 $\varepsilon$ 永远未被实测，且在审稿人眼中会造成“仅对基线/部署加噪、Stage 1 吃无噪数据”的公平性质疑。
  3. **$S_e$ 符号歧义与编码器 `init_bound` 硬截断风险**：L49 将 $S_e$ 写入部署公开常数 $\mathcal{C}$，L285 又用同一符号表示节点异质未知环境增益 $S_{e,i}\in[1.4,2.2]$。若将真实 $S_{e,i}$ 放入 $\mathcal{C}$ 即构成未知隐环境参数泄漏；若令公开常数 $S_e\equiv 1.0$，则需校核 `osgkd.py` 默认 `e_init_bound=2.0` 是否会在 `latent_overload` 强开放工况下硬截断真实初始环境转矩 $S_{e,i}e_{0,i}$。
  4. **Stage 1 内折 CV 泄漏与归一化常数口径**：4 折 CV 中若在切折前对全 `train` 做 $Q_\perp$ 与 $d_j$ 归一化会引入内折泄漏；Stage 2 的运动/端口归一化尺度（L257–258）未明文锁定“仅由 `train` 计算并冻结”。
- **最小必要修正方案**：
  - 仅按 $\mathcal{O}_1$ 可观测特征（训练轨线实测失稳结局 / 历史能量分位）做 CV 与批次分层；
  - 明确区分并分别命名“部署历史噪声”与“全轨迹/端口观测噪声”，至少在横轴增加一组全轨迹与 $\mathcal{O}_1$ 端口同时加噪的测试格以检验 Theorem 1 的 $\varepsilon$ 项；
  - 区分公开单位尺度 `S_e_scale = 1.0` 与生成器真参数 `env_gain_true`（严禁进入 $\mathcal{C}$），并依据 `train` $\mathcal{O}_1$ 端口分位数设定 `p/a/e_init_bound`；
  - 明确 Stage 1 4 折 CV 在每折 3/4 训练子集内独立拟合 $(\hat\vartheta, \hat{\tilde\gamma}, d_j)$ 并在验证折评估；明确 Stage 2 所有归一化常数仅用 `train` 冻结。

---

#### [M3] 外部标配基线（`NRI/ACD`、`PHNN/LGNN`、`SINDy-AE`）存在“草人基线”设计隐患

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` §4.4（L311–315, L320）。
- **技术证据**：
  - **`NRI / ACD`（L312）**：在单系统固定静态图任务中，若只喂 100 ms 的 $H_{100}$（11 帧）给边编码器推断二值图，而 PDD Stage 1 聚合全部 96 条 × 401 帧（4.0 s）训练轨迹，属明显不公平阉割；
  - **`PHNN / LGNN`（L315）**：仅描述 $H_\theta(\theta,\omega)+R_\theta\succeq 0+\text{外端口头}$，未界定 1,928 参编码器输出 $(p_0,a_0,e_0)$ 的演化方程；若做成 $2N$ 无隐态系统，在 $5N$ 开放系统上必然失效，极易被审稿人指为草人；
  - **`SINDy-AE`（L313, L320）**：在 256 步课程中每 32 步 STLSQ 硬剪枝，在 $T_s=1\ \mathrm{s}$ 阶段就剪枝两次，极易误杀慢开放项；且未说明自编码器潜坐标如何映射至显式边拓扑指标；
  - **`No-Privilege Same-Arch`（L311）**：未明确其同样享有纯运动一致性损失 $\mathcal{L}_{\mathrm{cons}}$；且 §1.2 中 $\mathcal{O}_0$ 定义仅为 100 ms 历史，建议单列 $\mathcal{O}_0^{\mathrm{traj}}$ 表示训练期无端口运动轨迹以消除符号混淆。
- **最小必要修正方案**：在进入实现前补齐一张各基线精确数学规格卡——允许 `NRI/ACD`（及 `NRI+Port`）在训练期吃 4 s 轨迹并跨轨迹池化系统级边概率；给出 `PHNN` 完整 $5N$ 开放端口哈密顿方程；将 `SINDy-AE` 的 STLSQ 置于密训预热（如 Phase B 末）之后并明文定义物理字典边提取规则；给 `No-Privilege` 同样开启 $\mathcal{L}_{\mathrm{cons}}$。

---

#### [M4] 单进程 CPU float64 算力预算存在汇总行笔误与两项低估（`PHNN` 二阶反传开销及 10 基线工程调试成本）

- **位置**：`reports/pdd_v3_architecture_20261128/scientist_design.md` §5.1（L340–359）。
- **复算与技术核查**：
  1. **汇总行算术笔误**：表 5.1 各行 ODE 运行数实为 `7 + 33 + 40 + 20 + 12 + 18 + 28 = 158` 条（与总更新数 `40,576 = 158 × 256 + 128` 一致），但 L354 误写为 **`128 ODE runs`**（漏写 30 条；同时未说明 Phase 1 锚格是否复用 Run 0 的 5 条种子 1 跑次）。
  2. **`PHNN / LGNN` 耗时与内存低估**：在 400 步 RK4 内每步 4 次 `autograd.grad(H, z, create_graph=True)` 叠加二阶反传，在 PyTorch CPU float64 下通常慢于 OSGKD **`4×–10×`**、峰值内存增 **`3×–5×`**，不能按 `0.132 h/run` 计（6 次运行实耗预计 `3–8 h`，须纳入 Run 0 dry-run 实测）。
  3. **`GraphNeuralODE` 双重 `_readout` 开销**：经查 `src/kiki/models/osgkd_baselines.py` L58，每次 `rhs` 均重复计算 `_readout(reference)`；在 $N=32$ 下建议单次 rollout 内按需缓存（数学严格等价），以防 $N=32$ 内存/耗时超限。
  4. **NeurIPS 2027 周期可行性判定**：即使计入 `PHNN`、`NRI`、新增 `No-Privilege Two-Stage` 与 2.5× 复跑余量，纯 CPU 计算墙钟约 **`100–180 h`（4–7.5 天）**，对约 20 周周期仍**完全可行**；真正的进度风险不是算力，而是 10+ 异构基线在 3 个物理系统上的无 bug 实现、零泄漏单测与公平调参（需明确预留 4–6 周工程窗口）。

---

### MINOR（写作准确性与治理细节提示）

#### [m1] `val_trajectory_audit.md` 与架构简报对 `neural_ode_24111` 5 条未达稳事件的排序表述不精确

- **位置**：`reports/pdd_v3_architecture_20261128/val_trajectory_audit.md` L3, L66；`designs/pdd_v3_architecture_brief.md` L21。
- **核查事实**：逐条对照 `val_audit_summary.json` 中 12 条可初始化失稳事件的 `true_t`，5 条 `window_no_los_unsettled` 事件为最快的 4 条（`0.14, 0.19, 0.21, 0.25 s`）加上**第 7 快**的 `pilot_0239`（`0.42 s`, 32 帧），而第 5、6 快的 `expansion_0186` 与 `pilot_0152`（均为 `0.33 s`, 23 帧）实际为 `window_settled`。应将“**恰为最快的 5 条故障**”修正为“**涵盖最快的 4 条及另 1 条较快故障 `pilot_0239`（`0.42 s`）**”。（其余全部 `val_trajectory_audit.md` 数值与 `val_audit_summary.json` 核对完全一致，且严格遵守零训练、零积分、零 `test` 访问边界。）

#### [m2] 评审镜像“字节一致”措辞偏差与 `gpt-6-astra` 暂定绑定治理闭环

- **位置**：`designs/pdd_v3_architecture_brief.md` L3, L5, L163, L208, L216；`results/pdd_v3_architecture_20261128/scientist_receipt.json`。
- **核查事实**：
  1. `designs/pdd_v3_architecture_brief.md`（`34,680 B`）L3 宣称与 `reports/pdd_v3_architecture_20261128/architecture_brief.md`（`34,922 B`）“字节一致”，实际相差 242 字节（4 处相对路径 `../` vs `../../` 及 CRLF/LF 差异，正文一致），宜改称“正文一致镜像（仅相对路径不同）”。
  2. `scientist_receipt.json` 如实留存了 `codex-cli 0.144.4` 调用 `gpt-6-astra` 返回 HTTP 400 的完整原始回执，无任何冒充；全部文件严格维持 `DRAFT / PAUSED_BY_USER / default none` 与红线（新 FOM = 0、IEEE-39 `test` 零访问、6 项“不做什么”护栏）。需同步更新简报中尚未闭合的“待 Scientist 二选一”状态，并在放行前就 `gpt-6-astra` 正式复核或暂定替代取得用户明确签批。

---

## 四、对 `root` 与用户的明确决策建议

1. **当前治理动作**：**维持 `DRAFT / PAUSED_BY_USER / default none`，暂不放行代码实现与 Run 0。**
2. **放行前必办修订包（按优先级，纯文档与静态代数核验，零训练/零积分）**：
   - **第一优先（修 B1）**：收紧 Theorem 1 结论 4 的 LUPI 理论口径（区分必然不可辨的端口分离规范 vs 需 $\tau_m \notin \mathcal{F}_{\mathrm{env}}$ / 含噪 / 非线性读出的骨架收益），增配 `No-Privilege Two-Stage` 对照，并同步调整基准 1 的噪声/驱动设定与 G3 门禁；
   - **第二优先（修 B2 + M1）**：修补 Theorem 1 的 $\tau_{\mathrm{env,cm}}(a)$ 速度泄漏边界、零共模 $\sin\theta_{\min}$ 前提、增广行权重一致性及分子 $N$ 缩放界，并补全 Theorem 1 结论 3 / Theorem 2 情形 1(b), 2 / Theorem 3 逃逸反例的遗漏前提；
   - **第三优先（修 B3）**：静态特征值重标基准 1 自然频带（或 Stage 1 通带），放宽 `Bounded-Recal` 投影盒至至少 $[0.55\hat c_e, 2.20\hat c_e]$ 并在 Run 0 与 `PDD-Frozen` 并列验证；静态不动点 + 雅可比特征根重标基准 3 Wilson–Cowan 四分块权重、STP 方向及 6 个缺失系数，并补齐基准 3 `Stage-1-Only` 定义；
   - **第四优先（修 M2 + M3 + M4 + m1–m2）**：剔除 $\mathcal{O}_2$ `excitation` 训练分层、明晰噪声与 $S_e$ 尺度口径、补全 `NRI/PHNN/SINDy-AE` 公平规格卡、更正表 5.1 的 `158` 运行数并将 `PHNN` 纳入 Run 0 dry-run，同时更正 `val_trajectory_audit.md` 中 `pilot_0239` 排序与简报“字节一致”措辞。
3. **需请用户明确拍板的治理事项**：
   - **事项 A（Scientist 模型绑定）**：按 `AGENTS.md` §4，请选择——① 升级本机 Codex CLI 至支持 `gpt-6-astra` 的版本，由真实 `gpt-6-astra` 复核修订后的 PDD v3 方案；或 ② 明示书面授权本轮在 `codex-cli` 不可用期间采纳修订后的暂定设计草案进入 Run 0（并在回执中继续保留暂定标注）。
   - **事项 B（修订闭环后放行 Run 0）**：上述 B1–B3 与 M1–M4 修订落盘并经一次快速复审确认后，即可安全放行进入最小判别 Run 0（基准 1 与基准 3 资格、零泄漏单测、含 `PHNN` 与 $N=32$ 的内存/耗时 dry-run 及 G1/G2 冒烟）。
