<!-- Host Receipt: results/pdd_v3_architecture_20261128/scientist_receipt.json -->
> **宿主调用真实回执（重要治理标注）**：
> 1. 本轮已实际尝试调用本机 `codex-cli 0.144.4` 执行 `gpt-6-astra`（`codex exec -m gpt-6-astra -s read-only --ephemeral --json`，线程 `01a221ac-432a-7780-9ac5-c6e8f770096d`），服务端返回 HTTP 400：`The 'gpt-6-astra' model requires a newer version of Codex. Please upgrade to the latest app or CLI and try again.`（完整真实回执见 [scientist_receipt.json](../../results/pdd_v3_architecture_20261128/scientist_receipt.json)）。
> 2. 严格遵守 `AGENTS.md` §4 与 `agents/scientist.md`（禁止冒充 `gpt-6-astra`），下列有界设计方案由上下文隔离的 Antigravity 只读 `research` 子代理（会话 `1639b919-2a4c-4032-b93d-7768bcefc975`，底层模型 `gemini-4-neon-xhigh`）起草，效力严格标定为 **`PROVISIONAL_DRAFT`（暂定草案）**，待用户在升级后的 Codex 宿主完成 `gpt-6-astra` 复核或明示批准等效替代绑定后转正。

# PDD v3（OSGKD-Decoupled）暂定 Scientist 有界设计方案

> **身份与暂定性质声明（严格遵循 `AGENTS.md` §4 与 `agents/scientist.md`）**
> - **角色身份**：暂定 Scientist 角色有界设计起草者（Provisional Scientist-Role Bounded Design Drafter），运行于 Antigravity 只读 research 子代理（会话 ID `1639b919-2a4c-4032-b93d-7768bcefc975`）。**本子代理不是、也严禁自称为 `gpt-6-astra`。**
> - **治理效力**：本稿为供 `root` 审阅、独立只读 Reviewer 挑战及用户决策的**暂定有界设计草案（Provisional Bounded Design）**；在真实 `gpt-6-astra` 审定签发或用户明示认可暂定替代前，不记为已完成 `gpt-6-astra` 正式回执。
> - **只读红线回执**：本轮严格只读检查了 10 份必读文件及对应依赖源码/报告（`osgkd_synthetic.py`、`negative_controls.py`、`osgkd.py`、`osgkd_graph.py`、`osgkd_baselines.py`、`osgkd_fitting.py`、`experiments/osgkd_v2_wp2_layer1_20261127/{models,engine,training}.py`、`escape_example.py`、`pendulum.py` 及 `wp2`/`wp3` 封存结果表）；**0 文件修改、0 训练更新、0 ODE 积分、0 `private_data/` 访问、0 IEEE-39 `test` 访问**。

---

## 0. Scientist 决策摘要（按 `agents/scientist.md` 必答要素拍板）

- **主假说**：在训练期具备特权聚合端口观测（$\mathcal{O}_1$）而部署期仅具短时部分运动历史（$\mathcal{O}_0$）的开放网络化动力学中，当环境闭合满足严格真因果限制 $\mathcal{F}_{\mathrm{env}}$ 且动态调制采用**奇对称有界核** $K_e(a)=c_e+\beta_e\tanh(a_i+a_j)$（$\kappa=0.8$）时，将**静态骨架恢复（Stage 1：零均值带通弱式 FISTA + 节点局部速度记忆去混叠）**与**固定骨架上的残差开放闭合及等变短历史部署初始化（Stage 2）**解耦，可在强开放/强调制工况下同时消除端到端跨模块互代、保住近精确拓扑恢复，并在部署期零特权输入条件下显著优于 `Stage-1-Only`、`End-to-End OSGKD v2`、`No-Privilege Same-Arch` 及黑箱连续 ODE 的大样本失稳分岔保真。
- **核心唯一拍板清单（零悬而未决选项）**：
  1. **动态调制核唯一拍板**：采用**奇对称有界核** $K_e(a) = c_e + \beta_e\tanh(a_i+a_j)$，$c_e\ge 0$，$0\le \beta_e \le 0.8\,c_e$（淘汰 $\exp(\tanh s_e)-1$，仅保留后者于 `End-to-End OSGKD v2` 历史基线内部以保持对旧实现保真）。
  2. **基准 3 唯一拍板**：选定**开放可塑 Wilson–Cowan 神经群体网络（Open Plastic Wilson–Cowan Neural Population Network）**，舍弃黏弹性非线性阻尼弹簧-质点网络。
  3. **Stage 1 / Stage 2 骨架规则拍板**：主方案严格冻结 $(\hat S, \hat c, \hat g)$（`PDD-Frozen`）；仅保留一项预登记消融 `Stage-2 Bounded-Recal`，将 $c_e$（$e\in\hat S_c$）限定于定理界定的投影盒 $[0.75\hat c_e,\,1.40\hat c_e]$ 内，严禁增删边或无约束覆盖。
- **竞争解释与最简对照**：① Stage 2 冗余（Weak SINDy 质疑）$\leftrightarrow$ 最简对照 `Stage-1-Only`（门禁 **G1**）；② 端到端联合拟合更优 $\leftrightarrow$ `End-to-End OSGKD v2`（门禁 **G2**）；③ 特权端口无净收益 $\leftrightarrow$ `No-Privilege Same-Arch`（门禁 **G3**）；④ 黑箱 ODE 在大样本下同样保持失稳或直接分类器即可替代 $\leftrightarrow$ `Equivariant Graph Neural ODE` / `Latent ODE` 与 `Direct Classifier` / `Seq2Seq`（门禁 **G4**）。
- **最有判别力的下一问题（Run 0）**：放行后先用 **`1,792` 次总更新（单进程 CPU 约 `1.4 h`）**在基准 1（$N=8$，`strong_open`）与基准 3（Wilson–Cowan $N=8$）上检验零泄漏资格、内存上限（至 $N=32$）以及 G1/G2 空间是否真正打开；若 Run 0 未在合成真值上拉开 `PDD` vs `Stage-1-Only` 的差距，立即止损修正工况或闭合，不盲目扩张至全十字星矩阵。
- **适用主张边界**：所有强机制与统计结论限于已声明的函数类 $\mathcal{F}_{\mathrm{env}}$、激励条件（A3）与公开合成基准及已封存 IEEE-39 探索性应用范围；不宣称跨领域冻结权重零样本迁移，不宣称高阶电力系统安全证书。

---

## 一、Theorem 1–3 精确数学陈述、假设检查与证明路线

> 目标篇幅：适配 NeurIPS 主文 `1.0–1.5` 页理论节（主文呈现精确定义、定理陈述与核心证明骨架，完整引理与常数推导放入附录 A）。

### 1.1 动态调制核唯一拍板与数学对比

| 核函数候选 | 显式表达与约束 | 范围与极值比 | 偶部系统偏差（对称分布 $s_e=a_i+a_j$） | Stage 2 冻结 $\hat c$ 兼容性 | Theorem 3 逃逸反例对照界 |
| --- | --- | --- | --- | --- | --- |
| 旧 v2 指数核（**淘汰**） | $c_e + \beta_e[\exp(\tanh s_e)-1]$，$0\le\beta_e\le c_e$ | $[0.368\,c_e,\,2.718\,c_e]$（不对称，$+172\% / -63\%$） | 偶部 $\cosh(\tanh s_e)-1\ge 0$ 严格非负；$\sigma_s\approx 1$ 时引入 $+21\%\text{–}35\%\,\beta_e$ 正向均值偏移 | **差**：Stage 1 估计出的 $\hat c_e$ 系统性高估 $c_e$，迫使冻结 $\hat c$ 的 Stage 2 给 $a(t)$ 注入永久负直流偏置以抵消偏差，重演跨模块补偿 | $0\le z(t)\le \frac{\pi e}{2}(1-e^{-t})\approx 4.270$ |
| **奇对称有界核（唯一拍板）** | **$K_e(a) = c_e + \beta_e\tanh(s_e)$，$c_e\ge 0$，$0\le\beta_e\le \kappa c_e$，$\kappa=0.8$** | **$[0.20\,c_e,\,1.80\,c_e]$（对称 $\pm 80\%$，峰谷比 **9×**）** | **严格奇函数** $\tanh(-s)=-\tanh(s)$；凡 $s_e$ 分布对称则一阶静态偏移**恒为零**（$\mu_e=\mathbb{E}[\tanh s_e]=0$） | **优**：Stage 1 估计无系统偶部偏移，$c_e^{\mathrm{eff}}\approx c_e$，主方案 `PDD-Frozen` 无直流补偿冲突；最坏非对称情形仍有严格双边正界 $[(1-\kappa)c_e,(1+\kappa)c_e]$ | **$0\le z(t)\le \frac{\pi\kappa}{2}(1-e^{-t}) < 1.257$**（更紧且证明仅 2 行） |

**拍板结论**：PDD v3 主干与所有新建合成基准真值**唯一采用奇对称有界核 $K_e(a) = c_e + \beta_e\tanh(a_i+a_j)$，固定 $\kappa = 0.8$**。
- **物理与储能势对接**：对应的一阶中心化互易势为
  $$U_c(\theta, a) = \sum_{e\in E}\Big[K_e(a)(1-\cos\varphi_e) - c_e(1-\cos\eta_e) - c_e\sin\eta_e(\varphi_e-\eta_e) - \beta_e(1-\cos\eta_e)s_e\Big],$$
  满足 $U_c(\theta_\star,0)=0$、$\nabla_\theta U_c = B[K(a)\odot\sin\varphi - c\odot\sin\eta] = F_c(\theta,a)$、$\nabla_a U_c\big|_{z_\star} = |B|[\beta\odot(\operatorname{sech}^2(0)(1-\cos\eta)-(1-\cos\eta))] = 0$，且 $\partial_{s_e} K_e(a) = \beta_e\operatorname{sech}^2(s_e) \in (0, 0.8 c_e]$ 全局有界光滑。
- **强失稳物理契合**：下界 $0.20 c_e$ 允许在大摆角电压跌落期将同步回复刚度削弱至标称值的 $20\%$（显著强于旧 v2 核的 $36.8\%$ 下界），同时严格远离零点保证全局正定回复与适定性。

---

### 1.2 观测算子层级与环境函数类 $\mathcal{F}_{\mathrm{env}}$ 的严格数学定义

设无向候选图 $G=(V,E)$，$|V|=N$，$|E|=m$（默认完全候选图 $m=\binom{N}{2}$），定向关联矩阵 $B\in\{-1,0,1\}^{N\times m}$（对边 $e=(i,j), i<j$ 取 $B_{i,e}=+1, B_{j,e}=-1$，故 $\mathbf{1}^\top B=0^\top$），无符号关联矩阵 $|B|\in\{0,1\}^{N\times m}$（满足 $\mathbf{1}^\top |B|=2\mathbf{1}_m^\top$）。系统完整状态为 $z(t)=(\theta(t),\omega(t),p(t),a(t),e(t))\in\mathbb{R}^{5N}$，$t\in[-h,T]$（$h=0.1\ \mathrm{s}, T=4.0\ \mathrm{s}$），参考平衡 $z_\star=(\theta_\star,0,0,0,0)$，$\varphi=B^\top\theta$，$\eta=B^\top\theta_\star$。令冻结公开常数集合 $\mathcal{C}=(M,D,\theta_\star,B,S_p,S_e,\omega_{\mathrm{scale}})$。

1. **严格嵌套观测算子 $\mathcal{O}_0 \subset \mathcal{O}_1 \subset \mathcal{O}_2$**：
   对轨迹集合 $\{z^{(r)}\}_{r=1}^R$：
   - **$\mathcal{O}_0$（部署级）**：$\mathcal{O}_0(z^{(r)}) := \big(\{(\theta^{(r)}(t_j),\omega^{(r)}(t_j))\}_{j=0}^{10},\; \mathcal{C}\big)$，$t_j=-h+j\Delta t\in[-0.1,0]\ \mathrm{s}$；
   - **$\mathcal{O}_1$（训练特权级）**：$\mathcal{O}_1(z^{(r)}) := \big(\mathcal{O}_0(z^{(r)}),\; \{(\theta^{(r)}(t),\omega^{(r)}(t),\tau_m^{(r)}(t),\tau_e^{(r)}(t))\}_{t\in[0,T]}\big)$，其中 $\tau_e = F_c(\theta,a) + F_{\mathrm{sym}}(\theta) + \tau_{\mathrm{env}}^{\mathrm{open}}(a,e)$ 为**聚合**总电气/交互端口；
   - **$\mathcal{O}_2$（神谕/冻结评分级）**：$\mathcal{O}_2(z^{(r)}) := \big(z^{(r)}(\cdot), F_c^{(r)}(\cdot), F_{\mathrm{sym}}^{(r)}(\cdot), \tau_{\mathrm{env}}^{\mathrm{open},(r)}(\cdot), S^\star, c^\star, g^\star, \beta^\star\big)$。
   显然生成的 $\sigma$-代数满足 $\sigma(\mathcal{O}_0)\subsetneq \sigma(\mathcal{O}_1)\subsetneq \sigma(\mathcal{O}_2)$。

2. **严格真因果开放环境函数类 $\mathcal{F}_{\mathrm{env}}(T_e, \Lambda_a, L_{\mathrm{cm}})$**：
   称开放环境端口 $\tau_{\mathrm{env}}^{\mathrm{open}}$ 属于 $\mathcal{F}_{\mathrm{env}}$，当且仅当：
   - **(E1 无瞬时差模 $\sin\varphi$ 直通)** $\tau_{\mathrm{env}}^{\mathrm{open}}(t) = S_e e(t) + \tau_{\mathrm{env,cm}}(a(t))$，满足 $\partial_\theta \tau_{\mathrm{env}}^{\mathrm{open}} \equiv 0$（严禁 v2 中的 `env_edges` 或任意含 $\sin\varphi$ 的瞬时边直通）；局部幅值读出 $\tau_{\mathrm{env,cm},i}(a) = \alpha_{1,i}\tanh(a_i) + \alpha_{2,i}\tanh^2(a_i)$ 全局 Lipschitz 常数不超过 $L_{\mathrm{cm}} = \max_i(|\alpha_{1,i}|+2|\alpha_{2,i}|)$；
   - **(E2 严格真因果一阶/有限阶记忆)** 隐态 $e(t), a(t)\in\mathbb{R}^N$ 遵循
     $$\dot e = -T_e^{-1} e + \Gamma_\omega \omega + \tilde\psi_e(\theta, a, e), \qquad \dot a = -\Lambda_a a + \psi_a(\theta, \omega, a),$$
     其中 $T_e^{-1}=\operatorname{diag}(T_{e,1}^{-1},\dots,T_{e,N}^{-1})\succ 0$（记 $T_{\min}^{-1}:=\max_i T_{e,i}^{-1}$）、$\Lambda_a\succ 0$（记 $\Lambda_{a,\max}:=\max_i \Lambda_{a,i}$），$\Gamma_\omega=\operatorname{diag}(\gamma_{e\omega,1},\dots,\gamma_{e\omega,N})$ 为节点局部对角速度耦合阵，且 $\tilde\psi_e,\psi_a$ 在轨线上平方可积（$\partial_\omega\tilde\psi_e\equiv 0$）。在频域上，算子 $(\frac{d}{dt}+T_{e,i}^{-1})^{-1}$ 具有严格真高频滚降 $|(i\Omega + T_{e,i}^{-1})^{-1}| = (\Omega^2 + T_{e,i}^{-2})^{-1/2} \le |\Omega|^{-1} = O(|\Omega|^{-1})\to 0$。

---

### 1.3 Theorem 1（特权端口 $\mathcal{O}_1$ 下的静态骨架可辨识性与函数类分离）

**数学陈述**：
设观测满足 **(A1)** $\mathcal{O}_1$ 聚合端口 $\tau_e = B[K(a)\odot\sin\varphi - c^\star\odot\sin\eta] + |B|[g^\star\odot(\cos\varphi-\cos\eta)] + \tau_{\mathrm{env}}^{\mathrm{open}}$（核取奇对称 $K_e(a)=c_e^\star+\beta_e^\star\tanh s_e$，$0\le\beta_e^\star\le\kappa c_e^\star$）；**(A2)** $\tau_{\mathrm{env}}^{\mathrm{open}}\in\mathcal{F}_{\mathrm{env}}$。

选取第 3.1 节构造的 $M$ 个零均值紧支导数调制带通测试函数 $\phi_k = \Psi'_k/\|\Psi'_k\|_{L^2(I_k)} \in C_c^1(I_k)$（对应原函数 $\Phi_k = \Psi_k/\|\Psi'_k\|_{L^2(I_k)}$ 满足 $\Phi_k(\partial I_k)=0$，中心角频率 $\Omega_k \ge \Omega_{\min} > 0$）。对 $R$ 条训练轨迹构建弱式堆叠向量 $y\in\mathbb{R}^{K_{\mathrm{tot}}}$（$K_{\mathrm{tot}}=RMN$）、静态设计阵 $\Phi_{\mathrm{stat}}=[\Phi_c,\Phi_g]\in\mathbb{R}^{K_{\mathrm{tot}}\times 2m}$ 及规范不变节点局部速度记忆干扰阵 $\Phi_{\mathrm{loc}}\in\mathbb{R}^{K_{\mathrm{tot}}\times N}$（第 $(r,k)$ 块为 $\operatorname{diag}(-\langle\Phi_k,\omega^{(r)}\rangle)$）。设正交投影 $Q_\perp = I - \Pi_{\operatorname{col}(\Phi_{\mathrm{loc}})}$（若 $\Gamma_\omega=0$ 则取 $Q_\perp=I$），记 $\tilde\Phi_{\mathrm{stat}}=Q_\perp \Phi_{\mathrm{stat}}$，归一化 Gram 阵 $\mathcal{G}=\frac{1}{K_{\mathrm{tot}}}\tilde\Phi_{\mathrm{stat}}^\top\tilde\Phi_{\mathrm{stat}}$。

假设满足 **(A3 宽带持续激励强制性)**：
1. 互易正弦块与全局求和余弦块分别满足 $\sigma_{\min}(\Phi_c/\sqrt{K_{\mathrm{tot}}})\ge \gamma_{\sin}>0$、$\sigma_{\min}(\Psi_{\cos}/\sqrt{RM})\ge \gamma_{\cos}>0$（其中 $(\Psi_{\cos})_{(r,k),e}=\langle\phi_k,\cos\varphi_e^{(r)}-\cos\eta_e\rangle$），且 $\|P_\perp\Phi_g/\sqrt{K_{\mathrm{tot}}}\|_2\le M_g<\infty$；
2. 静态列空间 $\mathcal{S}_{\mathrm{stat}}=\operatorname{col}(\Phi_{\mathrm{stat}})$ 与干扰空间 $\mathcal{S}_{\mathrm{loc}}=\operatorname{col}(\Phi_{\mathrm{loc}})$ 最小主夹角满足 $\theta_{\min}:=\angle(\mathcal{S}_{\mathrm{stat}},\mathcal{S}_{\mathrm{loc}})>0$（当 $\Gamma_\omega=0$ 时 $\sin\theta_{\min}=1$）。

则以下四项结论同时成立：

1. **结论 1（一般非二分图上的空间端点奇偶 × 函数奇偶联合分离与精确最小奇异值下界）**：
   即使在**一般非二分连通图**上 $\operatorname{col}(B)=\mathbf{1}^\perp \not\perp \operatorname{col}(|B|)=\mathbb{R}^N$，静态设计阵 $\Phi_{\mathrm{stat}}=[\Phi_c,\Phi_g]$ 仍具满列秩 $2m$，且投影后最小奇异值严格有正下界：
   $$\gamma_0 := \sigma_{\min}\!\left(\frac{\tilde\Phi_{\mathrm{stat}}}{\sqrt{K_{\mathrm{tot}}}}\right) \ge \gamma_{\mathrm{stat}}\sin\theta_{\min} > 0, \qquad \gamma_{\mathrm{stat}} := \sqrt{\min\!\left(\frac{2\gamma_{\cos}^2\gamma_{\sin}^2}{N^2 M_g^2 + 2\gamma_{\cos}^2},\; \frac{2}{N^2}\gamma_{\cos}^2\right)}.$$
   （若对全局求和行施加第 3.1 节的尺标均衡权重 $\sqrt{N/4}$，则下界中的 $N^2$ 改善为无 $N$ 退化的常数 $1$：$\gamma_{\mathrm{stat}}^W = \sqrt{\min\!\big(\frac{2\gamma_{\cos}^2\gamma_{\sin}^2}{4 M_g^2 + 2\gamma_{\cos}^2},\; \frac{1}{2}\gamma_{\cos}^2\big)}$。）特别地，在无调制无记忆子族下，$(c^\star,g^\star)$ 唯一可辨识。

2. **结论 2（瞬时直通 vs. 严格真因果记忆的精确高频投影分离与定量误差界）**：
   设 Stage 1 残差为 $r_{\mathrm{tot}} = Q_\perp(r_{\mathrm{mem}}^{\mathrm{rem}} + r_{\mathrm{mod}} + \varepsilon)$。首先，由规范不变分部积分恒等式 $\langle\phi_k, \theta_i\rangle = -\langle\Phi_k,\omega_i\rangle$，投影 $Q_\perp$ **精确消去**速度驱动记忆的 $O(1)$ 高频直通分量 $S_e\Gamma_\omega\theta$；剩余严格真因果记忆残差 $r_{\mathrm{mem}}^{\mathrm{rem}}$ 与局部幅值读出残差满足**显式有限区间衰减界**：
   $$\frac{\|r_{\mathrm{mem}}^{\mathrm{rem}}\|_2}{\sqrt{K_{\mathrm{tot}}}} \le \frac{1}{\Omega_{\min}}\Big[\|S_e\|_{\max}\big(T_{\min}^{-1}\bar E_e + \bar E_{\tilde\psi_e}\big) + L_{\mathrm{cm}}\big(\Lambda_{a,\max}\bar E_a + \bar E_{\psi_a}\big)\Big] = O\!\left(\frac{T_{\min}^{-1} + \Lambda_{a,\max} + 1}{\Omega_{\min}}\right),$$
   其中 $\bar E_x := \big(\frac{1}{K_{\mathrm{tot}}}\sum_{r,k}\|x^{(r)}\|_{L^2(I_k)}^2\big)^{1/2}$ 为每窗均方根能量。记剩余记忆残差与 $\operatorname{col}(\tilde\Phi_{\mathrm{stat}})$ 夹角为 $\theta_{\mathrm{rem}}$，则在无调制（$\beta^\star=0$）、有限观测噪声 $\varepsilon$ 下，无约束弱式最小二乘估计 $\hat\vartheta^{\mathrm{LS}}$ 满足：
   $$\|\hat\vartheta^{\mathrm{LS}} - \vartheta^\star\|_2 \le \frac{1}{\gamma_{\mathrm{stat}}\sin\theta_{\min}}\left[\frac{\cos\theta_{\mathrm{rem}}}{\Omega_{\min}}\Big(\|S_e\|_{\max}(T_{\min}^{-1}\bar E_e + \bar E_{\tilde\psi_e}) + L_{\mathrm{cm}}(\Lambda_{a,\max}\bar E_a + \bar E_{\psi_a})\Big) + \frac{\|\Pi_{\mathrm{stat}}\varepsilon\|_2}{\sqrt{K_{\mathrm{tot}}}}\right].$$
   **当 $\Omega_{\min}\to\infty$ 且无噪时，估计误差按 $\Omega_{\min}^{-1}$ 严格收敛至零。**

3. **结论 3（强奇对称调制下的有限样本 FISTA 误差界与支撑精确恢复）**：
   在强调制 $0\le\beta_e^\star\le\kappa c_e^\star$（$\kappa=0.8$）下，分解 $\tanh(s_e(t)) = \mu_e + \tilde h_e(t)$，则有效目标刚度 $c_e^{\mathrm{eff}} := c_e^\star + \beta_e^\star\mu_e \in [(1-\kappa)c_e^\star,(1+\kappa)c_e^\star] = [0.2 c_e^\star, 1.8 c_e^\star]$ 与 $c^\star$ **精确同支撑**（$c_e^{\mathrm{eff}}>0 \iff c_e^\star>0$），且在 $a(t)$ 符号对称激励下 $\mu_e=0$（**零系统偏差**）。对非负 $L_1$ 正则 Stage 1 估计 $\hat\vartheta = \arg\min_{\vartheta\ge 0}\frac{1}{2K_{\mathrm{tot}}}\|\tilde\Phi_{\mathrm{stat}}\vartheta - \tilde y\|_2^2 + \lambda_1\mathbf{1}^\top\vartheta$，只要 $\lambda_1 \ge \lambda_{\mathrm{crit}} := \big\|\frac{1}{K_{\mathrm{tot}}}\tilde\Phi_{\mathrm{stat}}^\top r_{\mathrm{tot}}\big\|_\infty$（含调制零均值波动项 $\tilde r_{\mathrm{mod}}$、记忆残差 $r_{\mathrm{mem}}^{\mathrm{rem}}$ 与噪声 $\varepsilon$），则对支撑大小 $s = |S^\star|$ 有：
   $$\|\hat\vartheta - \vartheta^{\mathrm{eff}}\|_2 \le \frac{2\lambda_1\sqrt{s}}{\gamma_0^2} = \frac{2\lambda_1\sqrt{s}}{\gamma_{\mathrm{stat}}^2 \sin^2\theta_{\min}}.$$
   特别地，若 $\frac{2\lambda_1\sqrt{s}}{\gamma_0^2} < \frac{1}{2}\min_{e\in S_c^\star} c_e^{\mathrm{eff}}$，则取阈值 $\tau_{\mathrm{supp}}\in\big(\frac{2\lambda_1\sqrt{s}}{\gamma_0^2},\, c_{\min}^{\mathrm{eff}}-\frac{2\lambda_1\sqrt{s}}{\gamma_0^2}\big)$ 保证**精确支撑恢复 $\hat S_c = S_c^\star$（$\mathrm{F1}=1.000$）**。

4. **结论 4（条件必要性）**：若取消 $\mathcal{O}_1$（退化至 $\mathcal{O}_0$）、取消 $\mathcal{F}_{\mathrm{env}}$ 允许瞬时差模 $\sin\varphi$ 直通、或取消 (A3) 宽带激励，则唯一可辨识性崩塌，精确等价类维数由 Theorem 2 给出。

**形式化证明路线（4 步无缝衔接，可直接排入 0.5 页证明骨架 + 附录 A）**：
- **Step 1（非二分图块三角下界，无需 $\operatorname{col}(B)\perp\operatorname{col}(|B|)$）**：
  利用 $\mathbb{R}^N = \operatorname{span}(\mathbf{1})\oplus \mathbf{1}^\perp$ 正交分解及 $\mathbf{1}^\top B = 0^\top$、$\mathbf{1}^\top |B| = 2\mathbf{1}_m^\top$，对任意 $(c,g)\in\mathbb{R}^{2m}$ 有精确勾股恒等式：
  $$\|\Phi_c c + \Phi_g g\|_2^2 = \frac{4}{N}\|\Psi_{\cos} g\|_2^2 + \|\Phi_c c + P_\perp \Phi_g g\|_2^2.$$
  （另注：逐边交叉块对角元 $(B^\top|B|)_{e,e}=(+1)(+1)+(-1)(+1)=0$ 对**任意图**恒为零，仅共顶点相邻边存在交叉。）对第二项应用精确 Young 恒等式 $\|u+v\|_2^2 = (1-\alpha)\|u\|_2^2 - \frac{1-\alpha}{\alpha}\|v\|_2^2 + \|\sqrt{\alpha}u+\frac{1}{\sqrt{\alpha}}v\|_2^2 \ge (1-\alpha)\|u\|_2^2 - \frac{1-\alpha}{\alpha}\|v\|_2^2$（$\alpha\in(0,1)$），取 $\frac{1-\alpha}{\alpha} = \frac{2\gamma_{\cos}^2}{N^2 M_g^2}$ 即刻得到 $\frac{1}{K_{\mathrm{tot}}}\|\Phi_c c+\Phi_g g\|_2^2 \ge \gamma_{\mathrm{stat}}^2(\|c\|_2^2+\|g\|_2^2)$。再由正交投影定义 $\|Q_\perp v\|_2 = \|v\|_2\sin\angle(v,\mathcal{S}_{\mathrm{loc}})\ge \|v\|_2\sin\theta_{\min}$，即得 $\gamma_0 \ge \gamma_{\mathrm{stat}}\sin\theta_{\min}$。（线性无关性核验：即使在小角线性化极限下，$L_c + \operatorname{diag}(\tilde\gamma) = 0$ 的非对角元 $-c_{ij}=0$ 立推 $c=0$ 进而 $\tilde\gamma=0$，故对所有 $N\ge 2$ 在持久激励下必有 $\sin\theta_{\min}>0$。）
- **Step 2（有限区间分部积分与精确 $\|\Phi_k\|_{L^2} < \Omega_k^{-1}$ 引理）**：
  对第 3.1 节构造的 $\Psi_{m,q}(u) = \sin^2(\pi u/L)\cdot\{\cos,\sin\}(\Omega_m u)$（$\Omega_m=2\pi m/L, m\ge 1$），$\Psi_{m,q}(0)=\Psi_{m,q}(L)=0$ 且 $\Psi'_{m,q}(0)=\Psi'_{m,q}(L)=0$。直接三角正交积分给出**精确闭式恒等式**：
  - 对 $m\ge 2$（余弦与正弦均同）：$\|\Phi_{m,q}\|_{L^2} = \frac{\|\Psi_{m,q}\|_{L^2}}{\|\Psi'_{m,q}\|_{L^2}} = \big(\Omega_m^2 + \frac{4\pi^2}{3L^2}\big)^{-1/2} < \Omega_m^{-1}$；
  - 对 $m=1$：$\|\Phi_{1,\cos}\|_{L^2} = \frac{\sqrt{7/8}}{\Omega_1} < \Omega_1^{-1}$，$\|\Phi_{1,\sin}\|_{L^2} = \frac{\sqrt{5/8}}{\Omega_1} < \Omega_1^{-1}$。
  将 $\phi_k = \Phi'_k$ 与 $e_i(t)$ 在 $I_k$ 上分部积分（边界项 $\Phi_k e_i\big|_{\partial I_k} = 0$ 精确消失）：
  $$\langle\phi_k, e_i\rangle = -\langle\Phi_k, \dot e_i\rangle = -\gamma_{e\omega,i}\langle\Phi_k,\omega_i\rangle + \langle\Phi_k, T_{e,i}^{-1}e_i - \tilde\psi_{e,i}\rangle.$$
  首项落入 $\mathcal{S}_{\mathrm{loc}}$ 被 $Q_\perp$ 精确零化；第二项与 $\langle\phi_k,\tau_{\mathrm{env,cm},i}(a)\rangle = -\langle\Phi_k,\tau'_{\mathrm{env,cm},i}(a)\dot a_i\rangle$ 直接由 Cauchy–Schwarz 与 $\|\Phi_k\|_{L^2}<\Omega_{\min}^{-1}$ 给出结论 2 的显式上界。
- **Step 3（非负凸锥变分不等式）**：
  由 $\hat\vartheta\ge 0$ 最优性及 $\vartheta^{\mathrm{eff}}\ge 0$ 可行性，$\langle \mathcal{G}(\hat\vartheta-\vartheta^{\mathrm{eff}}) - \frac{1}{K_{\mathrm{tot}}}\tilde\Phi_{\mathrm{stat}}^\top r_{\mathrm{tot}} + \lambda_1\mathbf{1},\; \vartheta^{\mathrm{eff}}-\hat\vartheta\rangle \ge 0$。当 $\lambda_1\ge\lambda_{\mathrm{crit}}$ 时，对所有 $j\notin S^\star$ 有 $\Delta_j = \hat\vartheta_j \ge 0$ 且 $(\frac{1}{K_{\mathrm{tot}}}\tilde\Phi_{\mathrm{stat}}^\top r_{\mathrm{tot}}-\lambda_1\mathbf{1})_j\le 0$，故非支撑和 $\le 0$；仅留支撑集上界 $2\lambda_1\|\Delta_{S^\star}\|_1 \le 2\lambda_1\sqrt{s}\|\Delta\|_2$，结合 $\Delta^\top\mathcal{G}\Delta \ge \gamma_0^2\|\Delta\|_2^2$ 立得结论 3 与精确支撑恢复推论。

---

### 1.4 Theorem 2（退化激励与无约束环境下的精确等价类维数）

设系统具有 $N$ 节点、$|E|$ 条候选边（$m=|S^\star|$ 条真实活动边），对应定向关联阵 $B\in\mathbb{R}^{N\times|E|}$ 及活动边关联阵 $A = B_{S^\star}\in\mathbb{R}^{N\times m}$。以下四类退化/放宽情形给出**精确矩阵秩与零空间维数公式**：

| 情形 | 退化/放宽假设 | 观测/灵敏度线性算子 $M$ | 精确秩 $\operatorname{rank}(M)$ | 精确等价类维数 $\dim\ker(M)$ | 显式等价类与 Layer 1 数值精确复现 |
| --- | --- | --- | :---: | :---: | --- |
| **1. 零角激励** | (1a) 同步平衡 $\theta(t)\equiv\theta_\star+\bar\theta\mathbf{1}, \omega\equiv p\equiv a\equiv e\equiv 0$（任意 $\theta_\star$）；或<br>(1b) 纯幅值初态 $a(0)\ne 0, \theta\equiv\theta_\star, \omega\equiv p\equiv e\equiv 0$（**须零流参考 $\eta=B^\top\theta_\star=0$**；若 $\eta\ne 0$ 则 $\beta\tanh(s)\sin\eta\ne 0$ 局部可观） | 灵敏度阵 $J_{(c,\beta)} = \frac{\partial(\tau_e,\dot z)}{\partial(c,\beta)} \equiv 0$（参数维数 $2|E|$） | **`0`** | **`2|E|`** | 任意 $(c,\beta)$ 给出完全相同的零运动右端与零端口。复现 Layer 1（$N=8,|E|=28$）：$2|E|=56 \Rightarrow$ **秩 `0`、零空间 `56`**。 |
| **2. 单频/同频补偿** | 活动边角波形与环境共享单一非零时间波形 $f(t)$（$T_s$ 采样帧向量 $\mathbf{f}\ne 0$）：$y(t)=(Ac+q)f(t)$，$\xi=(c,q)\in\mathbb{R}^{m+N}$ | Kronecker 积算子 $M = \mathbf{f}\otimes [A, I_N]\in\mathbb{R}^{(T_s N)\times(m+N)}$ | **`N`**（因 $\operatorname{rank}(\mathbf{f})=1$ 且 $\operatorname{rank}([A,I_N])=N$） | **`m`**（即 $(m+N)-N$） | $\ker(M)=\{(\Delta c,\,-A\Delta c):\Delta c\in\mathbb{R}^m\}$。取 $\Delta c=-c_{e_0}^\star\mathbf{e}_{e_0}$ 可将 $m$ 条真边合法删减为 $m-1$ 条而观测严格不变。复现 Layer 1（$N=8,m=12$）：参数 $20 \Rightarrow$ **秩 `8`、零空间 `12`**（12 边 $\to$ 11 边，观测差 $<10^{-14}$）。 |
| **3. 无约束瞬时差模环境混叠** | 保留 $\mathcal{O}_1$ 与满宽带激励 (A3)，但取消 $\mathcal{F}_{\mathrm{env}}$，允许 $\tau_{\mathrm{env}}^{\mathrm{inst}}=B[q_{\mathrm{env}}\odot(\sin\varphi-\sin\eta)]$（或任意边特征直通） | $\Phi_c(c+q_{\mathrm{env}}) = [\Phi_c,\Phi_c]\begin{bmatrix}c\\ q_{\mathrm{env}}\end{bmatrix}$ | $\le |E|$（对 $(c,q_{\mathrm{env}})\in\mathbb{R}^{2|E|}$） | **$\ge |E|$**（放开 $a$ 调制则 $\ge 2|E|$） | $\{(c+\Delta,\,q_{\mathrm{env}}-\Delta): \Delta\in\mathbb{R}^{|E|}, c+\Delta\ge 0\}$。因 $P_\perp B=B$，零和投影 $P_\perp$ **完全无法**消去此等价类。 |
| **4. $\mathcal{O}_0$ 端口平移规范** | 退化为纯运动观测 $\mathcal{O}_0$（无 $\tau_m,\tau_e$ 独立观测），动量方程仅暴露净力 $\tau_{\mathrm{net}}=\tau_m-\tau_e$ | 端口差映射 $(\tau_m,\tau_e)\mapsto \tau_m-\tau_e$ | 每帧 $N$（总自由度 $2N$） | **逐帧 $N$ 维**（定常/共享衰减模态参数空间至少 **$N$ 维**） | $(\tau_m(t),\tau_e(t))\mapsto (\tau_m(t)+\delta(t),\,\tau_e(t)+\delta(t))$ 保持 $(\theta,\omega)$ 严格不变；若机械驱动未限为节点局部，则还可吸收任意 $B[\Delta\odot\sin\varphi]$ 使静态骨架亦不可辨。 |

**证明步骤**：情形 1 直接由 $K_e(0)\sin\eta_e - c_e\sin\eta_e \equiv 0$ 及 $\eta=0\Rightarrow\sin\varphi_e\equiv 0$ 逐项求导得零矩阵；情形 2 由 Kronecker 积秩公式 $\operatorname{rank}(\mathbf{f}\otimes[A,I_N])=\operatorname{rank}(\mathbf{f})\cdot\operatorname{rank}([A,I_N])=1\cdot N$ 及秩-零度定理直接得出，且非负支撑缩减见证由 $\Delta c = -c_{e_0}^\star\mathbf{e}_{e_0}$ 构造验证；情形 3 由 $[I_{|E|}, I_{|E|}]$ 零空间维数 $|E|$ 及 $P_\perp B=B$ 给出；情形 4 由 $[I_N, -I_N]$ 零空间 $\{(v,v):v\in\mathbb{R}^N\}$ 给出。

---

### 1.5 Theorem 3（奇对称有界核开放闭合的全局适定性与有限时间逃逸反例）

**数学陈述**：
1. **全局存在唯一性（对接奇对称有界核）**：对固定有限图 $G=(V,E)$、$M\succ 0$、$D\succeq 0$、有限正尺度 $(S_p,S_e,\omega_{\mathrm{scale}})$、奇对称有界核 $K_e(a)=c_e+\beta_e\tanh(a_i+a_j)$（$c_e\ge 0, 0\le\beta_e\le\kappa c_e, \kappa=0.8$）及第 3.2 节有界非线性字典（所有非线性基函数为 $\sin,\cos,\tanh,\tanh^2$，线性项系数有限），完整 $5N$ 维向量场 $f:\mathbb{R}^{5N}\to\mathbb{R}^{5N}$ 为 $C^\infty$ 局部 Lipschitz（本具体无状态乘积闭合甚至全局 Lipschitz）且满足显式全局线性增长界：
   $$\|f(z)\|_2 \le A + L\|z\|_2 \qquad (\forall z\in\mathbb{R}^{5N}),$$
   其中
   $$A = \|M^{-1}\|_2\Big(2(1+\kappa)\|B\|_2\|c\|_2 + 2\||B|\|_2\|g\|_2 + \|\alpha_1\|_2 + \|\alpha_2\|_2\Big) + A_{\mathrm{lat,bdd}},$$
   $$L = 1 + \|M^{-1}\|_2\big(\|FS_p\|_2 + \|D\|_2 + \|S_e\|_2\big) + L_{\mathrm{lat,lin}}.$$
   由 Picard–Lindelöf 局部存在唯一性与 Grönwall 不等式 $\|z(t)\|_2 \le (\|z(0)\|_2 + At)e^{Lt}$，任意有限初态 $z(0)\in\mathbb{R}^{5N}$ 的唯一解延拓至所有有限时刻 $t\in[0,\infty)$，**严格排除有限时间逃逸**。（基准 3 的 $4N$ 维 Wilson–Cowan 向量场同理满足 $[0,1]^N$ 前向不变性与全局线性增长，拥有完全同构的全局存在唯一性。）
2. **两机无界核有限时间逃逸解析反例（结构必要性）**：在已独立复核的两机储能不变子系统 $\dot z = \frac{\pi}{2}K'(z)-z,\; z(0)=0$ 上：
   - **无界指数核** $K(z)=e^z$：对 $z\ge 0$ 有 $e^z\ge 1+z>z$，故 $\dot z\ge (\frac{\pi}{2}-1)e^z$，分离变量得有限爆破上界 $T_{\mathrm{esc}} \le (\frac{\pi}{2}-1)^{-1} \approx 1.751938\ \mathrm{s}$（与 `escape_example.py` 一致：到达 $z=16$ 时刻 `0.761924409 s`，解析尾界 $1.972\times 10^{-7}\ \mathrm{s}$）；
   - **选定奇对称有界核** $K^{\mathrm{odd}}(z)=1+\kappa\tanh z$（$\kappa=0.8$）：$K'(z)=\kappa\operatorname{sech}^2 z\in(0,\kappa]$，由 $\dot z(0)=\frac{\pi\kappa}{2}>0$ 得 $z(t)\ge 0$ 且 $\dot z + z \le \frac{\pi\kappa}{2}$，积分因子 $e^t$ 立得全局一致有界：
     $$0 \le z(t) \le \frac{\pi\kappa}{2}(1-e^{-t}) < \frac{\pi\kappa}{2} \approx 1.2566 \qquad (\forall t\ge 0).$$
   （历史 v2 有界核 $e^{\tanh z}$ 对应上界 $\frac{\pi e}{2}(1-e^{-t})$ 保持完全兼容并作为参照并列复核。）

**明确不蕴含的结论（防夸大红线）**：
Theorem 3 **不蕴含**：① 长期一致有界性（$\sup_{t\ge 0}\|z(t)\|<\infty$，线性增长界允许且物理上必须允许真实失稳轨线指数分离）；② 局部/全局渐近稳定性、被动性或 Lyapunov 吸引域（RoA）证书；③ 显式定步长 RK4 在任意大步长 $\Delta t$ 下的数值稳定性（仍须逐格通过双步长资格核验）；④ 底层高阶 FOM（如 IEEE-39）的适定性或安全证书。

---

## 二、基准 3（非振子开放网络系统）唯一最终选型与完整规格

### 2.1 唯一二选一拍板与五维淘汰论证

**唯一拍板选定：开放可塑 Wilson–Cowan 神经群体网络（Open Plastic Wilson–Cowan Neural Population Network）**；**明确舍弃：开放黏弹性非线性阻尼弹簧-质点网络**。

| 评估维度 | 开放 Wilson–Cowan 神经群体网络（**入选**） | 开放黏弹性非线性阻尼弹簧-质点网络（**舍弃**） |
| --- | --- | --- |
| **1. 与振子方程族的本质数学差异** | **本质跨方程族**：一阶非哈密顿耗散发放率动力学（无惯量 $M$、无二阶动量、无辛结构）、遵循 Dale 法则的**有向非互易** E/I 突触耦合、单调饱和 Sigmoid 激活与有界状态空间 $[0,1]^N$。彻底击碎“只是换皮摇摆方程”的审稿质疑。 | **实质同族**：仍为二阶牛顿力学网络 $M\ddot q+D\dot q+B[k\odot\phi(B^\top q)]=f$，与基准 1（带惯性 Kuramoto）和基准 2（机械摆阵）同属二阶互易势网络，跨方程族说服力不足。 |
| **2. $\mathcal{O}_1$ 特权端口与 $\mathcal{O}_0$ 部署观测的物理自然性** | **教科书级 LUPI 映射**：训练期 $\mathcal{O}_1$ 为侵入式胞内/膜片钳记录的聚合突触与神经调质净电流 $(I_{\mathrm{drive}}, I_{\mathrm{net}})$；部署期 $\mathcal{O}_0$ 仅为非侵入式宏观群体发放率短时历史 $r(t)$（钙成像/多单元记录）。 | 与基准 2 机械摆阵（实验室力矩传感器 vs 外场位速）在物理叙事上高度重复，缺乏独立跨领域支撑。 |
| **3. Theorem 1 分离条件的直接对应性** | **精准同构且无需互易性**：瞬时有向突触骨架 $W(a)(r-r_\star)$ vs 严格真因果慢适应/星形胶质环境记忆 $m(t)$（$\dot m$ 仅由 $r,a,m$ 驱动、不含 $\dot r$，100% 严格真 $O(\Omega_m^{-1})$ 衰减！）+ 短时突触可塑性（STP）奇对称调制；$\mathcal{O}_0$ 下精确存在同构的 $N$ 维电流平移规范 $(I_{\mathrm{drive}}+\delta, I_{\mathrm{net}}-\delta)$。 | 仅复述互易 $B$ 与速度阻尼结构，未能检验两阶段解耦脱离 $B/\sin$ 后的可迁移性。 |
| **4. 定性失稳/分岔是否清晰可测** | **典型双稳态鞍结癫痫跃迁（Bistable Seizure Runaway）**：低发放稳态（$r_\star\approx 0.15$）与高发放超同步癫痫吸引子（$\bar r_E > 0.65$）由清晰分界面隔开；短时突触易化 $a$ 增强 $W_{EE}$ 与慢适应耗竭决定是否越过分界，与振子失步互补。 | 黏弹性弹簧-质点通常为单稳态耗散（除非人工构造负刚度屈曲），失稳边界不如双稳态神经癫痫跃迁自然清晰。 |
| **5. 单进程 CPU 算力成本** | 状态维数仅 **$4N$**（$N=8$ 为 32 维，比振子 40 维更轻），$\tau_r\in[0.08,0.15]\ \mathrm{s}$ 无刚性，RK4 $\Delta t=0.01\ \mathrm{s}$ 稳定且单步耗时较基准 1 低约 **20%–25%**。 | 状态维数 $5N$ 且高刚度黏弹性项易引入数值刚性（增大步长资格负担）。 |

### 2.2 开放可塑 Wilson–Cowan 网络完整数学规格

- **网络拓扑与状态维数（$4N$ 维，$N\in\{8, 16\}$）**：
  - 遵循生物 Dale 法则：将节点分为已知类型的兴奋性（E）与抑制性（I）群体，记已知冻结符号 $\sigma_j=+1\ (j\in V_E)$、$\sigma_j=-1\ (j\in V_I)$。$N=8$ 取 $6\text{E}+2\text{I}$；$N=16$ 取 $12\text{E}+4\text{I}$。候选图为有向无自环完全图（$|E_{\mathrm{dir}}|=N(N-1)$：$56$ 条于 $N=8$，$240$ 条于 $N=16$），真实稀疏突触支撑 $|S^\star| \approx 2.5N$（$20/56$ 于 $N=8$，$40/240$ 于 $N=16$，含 E$\to$E、E$\to$I、I$\to$E、I$\to$I 环路）。
  - 完整状态 $z = (r, u, a, m) \in [0,1]^N \times \mathbb{R}^N \times \mathbb{R}^N \times \mathbb{R}^N$：
    - $r_i \in [0, 1]$：可观群体发放率；
    - $u_i$：不可观外源丘脑/皮层下主动驱动电流（对应振子 $p_i$）；
    - $a_i$：不可观短时突触可塑性（STP 易化/抑制）神经调质隐态（对应振子 $a_i$）；
    - $m_i$：不可观慢发放频率适应 / 胞外离子环境记忆（对应振子 $e_i$）。
- **已知硬生物物理先验（冻结公开常数）**：
  - 膜时间常数 $\tau_{r,i}\in[0.08,0.15]\ \mathrm{s}$、不应期系数 $\alpha_r = 0.15$（稳态上限 $r_{\max}=1/1.15\approx 0.870$）、单调传递函数 $S(x) = (1 + \exp(-\gamma_s(x - x_0)))^{-1}$（$\gamma_s=5.0, x_0=0.5$）、已知细胞类型符号 $\sigma_j\in\{+1,-1\}$、已知稳态参考发放率 $r_{\star,i}\in[0.14,0.18]$ 及对应平衡偏置电流 $I_{0,i} := S^{-1}\!\big(\frac{r_{\star,i}}{1-\alpha_r r_{\star,i}}\big)$（保证在 $z_\star=(r_\star,0,0,0)$ 处 $\dot r\equiv 0$ 精确成立）。
- **完整微分方程**：
  $$\tau_{r,i}\dot r_i = -r_i + (1 - \alpha_r r_i)\,S\!\big(I_{0,i} + I_{\mathrm{drive},i} + I_{\mathrm{net},i}\big),$$
  $$I_{\mathrm{drive},i} = u_i, \qquad I_{\mathrm{net},i} = \underbrace{\sum_{j\ne i} \sigma_j\, w_{ij}(a)\,(r_j - r_{\star,j})}_{F_{\mathrm{syn},i}(r,a)} \;\underbrace{- S_{m,i} m_i + \alpha_{\mathrm{cm},i}\tanh(a_i)}_{I_{\mathrm{env},i}^{\mathrm{open}}(a,m)},$$
  其中奇对称有界 STP 调制核（与基准 1 **完全同构**）：
  $$w_{ij}(a) = w_{ij} + \beta_{ij}\tanh(a_i + a_j), \qquad w_{ij}\ge 0,\quad 0\le \beta_{ij}\le 0.8\,w_{ij}.$$
  隐态方程（严格真因果，$\partial_{\dot r}\psi_m \equiv 0$）：
  $$\dot u_i = -\lambda_{u,i} u_i - \gamma_{ur,i}(r_i - r_{\star,i}), \qquad \lambda_{u,i}\in[0.45,0.75]\ \mathrm{s}^{-1},$$
  $$\dot a_i = -\lambda_{a,i} a_i + \gamma_{ar,i}(r_i - r_{\star,i}) - \gamma_{a,\mathrm{sat},i}\tanh(a_i), \qquad \lambda_{a,i}\in[0.40,0.65]\ \mathrm{s}^{-1},$$
  $$\dot m_i = -T_{m,i}^{-1} m_i + \gamma_{mr,i}(r_i - r_{\star,i}) + \gamma_{ma,i}\tanh(a_i), \qquad T_{m,i}^{-1}\in[0.20,0.40]\ \mathrm{s}^{-1}.$$
  （物理强开放/强调制参数：静态 E 权行和 $\approx 1.7\text{–}2.1$、I 权行和 $\approx 1.1\text{–}1.5$ 保证低发放平衡 $r_\star$ 局部渐近稳定；$2/3$ 活动突触具强调制 $\beta_{ij}\in[0.65,0.80]w_{ij}$，且 $S_{m,i}\in[1.2,1.9]$，使调制电流 RMS 占 $I_{\mathrm{net}}$ 的 $20\%\text{–}35\%$、慢适应环境电流 RMS 占 $35\%\text{–}50\%$。）
- **三级观测算子 $\mathcal{O}_0 / \mathcal{O}_1 / \mathcal{O}_2$**：
  - $\mathcal{O}_0$：11 帧 100 ms 因果发放率历史 $\{r(t_j)\}_{j=0}^{10}$（$t_j\in[-0.1,0]\ \mathrm{s}$）+ 冻结公开常数 $(\tau_r, \alpha_r, \gamma_s, x_0, \sigma, r_\star, I_0)$；**部署期严禁输入任何电流或隐态**。
  - $\mathcal{O}_1$：训练期观测轨迹 $r(t)$ + 聚合特权电流 $(I_{\mathrm{drive}}(t), I_{\mathrm{net}}(t))$（$t\in[0,4.0]\ \mathrm{s}$）。
  - $\mathcal{O}_2$：真实分离电流 $(F_{\mathrm{syn}}, I_{\mathrm{env}}^{\mathrm{open}})$、真有向突触阵 $(S^\star, w^\star, \beta^\star)$ 与隐态 $(u, a, m)$（仅冻结评分）。
- **定性失稳/双稳态癫痫跃迁判据**：
  - 稳态（homeostatic）：尾窗 $[3.0, 4.0]\ \mathrm{s}$ 上所有帧满足兴奋性群体均发放率 $\bar r_E(t) := \frac{1}{|V_E|}\sum_{i\in V_E} r_i(t) < 0.35$。
  - 失稳/癫痫态转换（seizure transition）：在 $t\in(0, 4.0]\ \mathrm{s}$ 内存在连续 20 帧（$0.20\ \mathrm{s}$）满足 $\bar r_E(t) > 0.65$（且因果历史窗 $[-0.1,0]\ \mathrm{s}$ 内全程 $\bar r_E < 0.50$ 以确保 100% 因果可初始化）。连续 AUROC 评分量取 $s_{\mathrm{seiz}} = \max_{t\in[0,4.0]} \bar r_E(t)$。
- **Stage 1 / Stage 2 对应形式**：
  - **Stage 1**：因 $\dot m$ 无 $\dot r$ 驱动，直接用与基准 1 **完全相同**的 48 个零均值带通测试函数 $\phi_k$ 对 $I_{\mathrm{net},i}(t)$ 投影，分解为 $N$ 个独立的 $(N-1)$ 维非负凸 FISTA 问题：$\min_{w_{i,\cdot}\ge 0} \frac{1}{2RM}\sum_{r,k}\big(\langle\phi_k, I_{\mathrm{net},i}^{(r)}\rangle - \sum_{j\ne i} \sigma_j w_{ij}\langle\phi_k, r_j^{(r)}-r_{\star,j}\rangle\big)^2 + \lambda_1\sum_{j\ne i} d_{ij} w_{ij}$，再用 4 折内部 1-SE CV 选 $\lambda_1$、$\tau_{\mathrm{supp}}=0.05$ 定支撑 $\hat S$ 并去偏 NNLS 精修 $\hat w$。
  - **Stage 2**：冻结 $(\hat S, \hat w)$，仅在 $\hat S$ 上学 $\beta_{ij}\in[0,0.8\hat w_{ij}]$、严格真残差动力学 $(\dot u, \dot a, \dot m)$ 及同构等变图 GRU 发放率编码器 $I_\phi^{\mathrm{WC}}$（以 $(r_i-r_{\star,i}, \sigma_i, \log\tau_{r,i})$ 及节点对差 $(r_i-r_j, \sigma_j)$ 为唯一 $\mathcal{O}_0$ 合法输入，扣除参考平衡读出，严格保持末帧 $r(0)$，参数量 `~1,864`）。

---

## 三、Stage 1 / Stage 2 精确算法接口、损失与零泄漏规范

### 3.1 Stage 1：带通弱式测试函数、非负 FISTA 与对称损耗联合接口

1. **显式光滑紧支零均值带通测试函数族 $\phi_k(t)$**（纠正旧 Layer 1 `weak_force_fit` 采用低通正窗导致慢记忆 $S_e e(t)$ 100% 穿透的根因缺陷）：
   - 在 $T=4.0\ \mathrm{s}$、采样间隔 $\Delta t=0.01\ \mathrm{s}$（$401$ 帧）上构造多尺度紧支窗：
     - **尺度 1（高通主力）**：$W_1=7$ 个半重叠 $L_1=1.0\ \mathrm{s}$ 子窗（$[0,1],[0.5,1.5],\dots,[3,4]\ \mathrm{s}$，每窗 $N_w=100$ 区间），每窗取谐波 $m\in\{1,2,3\}$（角频率 $\Omega_m = 2\pi, 4\pi, 6\pi\ \mathrm{rad/s}$，对应 $1, 2, 3\ \mathrm{Hz}$）$\times$ 正余弦双正交相位 $q\in\{\cos,\sin\}$，共 $7\times 6 = 42$ 个测试函数；
     - **尺度 2（中频补充）**：$W_2=3$ 个半重叠 $L_2=2.0\ \mathrm{s}$ 子窗（$[0,2],[1,3],[2,4]\ \mathrm{s}$，$N_w=200$ 区间），取 $m=1$（$\Omega=\pi\ \mathrm{rad/s}=0.5\ \mathrm{Hz}$）$\times\{\cos,\sin\}$，共 $6$ 个测试函数；
     - 合计每条轨迹 **$M = 48$ 个测试函数**，通带严格覆盖 $[0.5, 3.0]\ \mathrm{Hz}$（$\Omega_{\min}=\pi\ \mathrm{rad/s}$，远高于开放记忆截止频率 $T_e^{-1}\le 0.40\ \mathrm{s}^{-1}$，衰减因子 $T_e^{-1}/\Omega_m\in[0.01,0.13]$）。
   - 解析构造与原函数：对子窗 $I_w=[t_w, t_w+L]$ 上局部时间 $u=t-t_w\in[0,L]$，定义
     $$\Psi_{m,\cos}(u) = \sin^2\!\left(\frac{\pi u}{L}\right)\cos(\Omega_m u), \qquad \Psi_{m,\sin}(u) = \sin^2\!\left(\frac{\pi u}{L}\right)\sin(\Omega_m u),$$
     $$\phi_{m,q}(u) := \frac{\Psi'_{m,q}(u)}{\|\Psi'_{m,q}\|_{L^2(0,L)}}, \qquad \Phi_{m,q}(u) := \frac{\Psi_{m,q}(u)}{\|\Psi'_{m,q}\|_{L^2(0,L)}}.$$
   - **$\Delta t=0.01\ \mathrm{s}$ 离散步长兼容性**：最高谐波频率为 $(m+1)/L = 4\ \mathrm{Hz}$（Nyquist 为 $50\ \mathrm{Hz}$，每周期 $25$ 个采样点）。因 $\Psi_{m,q}$ 与 $\phi_{m,q}$ 均在端点零化且为有限阶三角多项式，复合梯形求积对其积分为**谱精确（误差 $<10^{-15}$）**；离散权重向量 $\mathbf{w}_k$ 在内部节点强制扣除均值至 $10^{-16}$ 并归一为 $\|\mathbf{w}_k\|_2=\sqrt{\Delta t}$，离散分部积分与连续 Theorem 1 界偏差 $\le O(\Delta t^2\Omega_{\max}^2) < 3.5\%$。

2. **对称损耗联合估计与节点局部速度记忆去混叠接口**：
   - 对每条训练轨迹 $r$ 与每个测试函数 $k$ 计算：
     - 目标 $y_{r,k} = \langle\phi_k,\tau_e^{(r)}\rangle\in\mathbb{R}^N$；
     - 互易块 $\Phi_{c,(r,k)} = B\operatorname{diag}(\langle\phi_k,\sin\varphi^{(r)}-\sin\eta\rangle)\in\mathbb{R}^{N\times|E|}$；
     - 对称损耗块（按工况/模型开关启用）$\Phi_{g,(r,k)} = |B|\operatorname{diag}(\langle\phi_k,\cos\varphi^{(r)}-\cos\eta\rangle)\in\mathbb{R}^{N\times|E|}$；
     - **全局求和尺标均衡增广行**（落定 Theorem 1 机制 1）：追加标量行 $\sqrt{\frac{1}{4}}\mathbf{1}^\top y_{r,k} = \sqrt{\frac{1}{4}}\cdot 2\sum_e g_e\langle\phi_k,\cos\varphi_e^{(r)}-\cos\eta_e\rangle + \cdots$（对所有 $N\in\{8,16,32\}$ 保持全局求和通道权重与 $N$ 无关）；
     - **规范不变节点局部速度记忆干扰块**（基准 1/2 启用，基准 3 省略）：$\Phi_{\mathrm{loc},(r,k)} = \operatorname{diag}(-\langle\Phi_k,\omega^{(r)}\rangle)\in\mathbb{R}^{N\times N}$。因干扰参数 $\tilde\gamma\in\mathbb{R}^N$ 无约束且不加惩罚，先用精确 Schur 补 / Frisch–Waugh–Lovell 投影 $Q_\perp = I - \Phi_{\mathrm{loc}}(\Phi_{\mathrm{loc}}^\top\Phi_{\mathrm{loc}}+10^{-12}I_N)^{-1}\Phi_{\mathrm{loc}}^\top$（仅需 $N\times N$ 线性求解）解析消去 $\tilde\gamma$，得到降噪后设计阵 $\tilde\Phi_{\mathrm{stat}}=Q_\perp\Phi_{\mathrm{stat}}$ 与目标 $\tilde y=Q_\perp y$。

3. **非负重启 FISTA、训练内部 4 折 1-SE 选参、支撑阈值与去偏 NNLS**：
   - 列归一化非负凸优化目标（记列 RMS $d_j = \|\tilde\Phi_{\mathrm{stat},\cdot,j}\|_2/\sqrt{K_{\mathrm{tot}}}$）：
     $$\min_{\vartheta=(c,g)\ge 0}\; \frac{1}{2K_{\mathrm{tot}}}\|\tilde\Phi_{\mathrm{stat}}\vartheta - \tilde y\|_2^2 + \lambda_1 \sum_j d_j \vartheta_j.$$
   - **求解器**：O'Donoghue–Candès 自适应重启非负 FISTA，精确步长 $1/\lambda_{\max}(\mathcal{G})$，上限 `2000` 步，投影 KKT 残差 $\le 10^{-10}$ 收敛（$N=32$ 耗时 $<0.5\ \mathrm{s}$）。
   - **训练内部 4 折 1-SE 规则选 $\lambda_1$（零 `val`/`test` 接触）**：在 `train` 内部按激励/失稳类别分层切 4 折；搜索网格 $\lambda_1 / \lambda_{\max}^{\mathrm{data}} \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}, 3\times 10^{-2}, 10^{-1}\}$（其中 $\lambda_{\max}^{\mathrm{data}}=\max_j \frac{1}{K_{\mathrm{tot}}d_j}|\tilde\Phi_{\cdot,j}^\top\tilde y|$），选取 CV 弱式 MSE 落在最低均值 $+1\ \mathrm{SE}$ 内的**最大** $\lambda_1^{\mathrm{1SE}}$。
   - **支撑判定与去偏 refit**：在全量 `train` 用 $\lambda_1^{\mathrm{1SE}}$ 拟合 $\hat\vartheta^{\mathrm{FISTA}}$；连续排序分（供 AUROC/AUPRC 评分）取 $s_e^{\mathrm{score}} = d_{c,e}\hat c_e^{\mathrm{FISTA}}$；硬支撑由相对弱式功能 RMS 占比 $\rho_e^{\mathrm{func}} = \frac{d_{c,e}\hat c_e^{\mathrm{FISTA}}}{\max_{e'} d_{c,e'}\hat c_{e'}^{\mathrm{FISTA}}} \ge \tau_{\mathrm{supp}}$ 确定，**预登记固定 $\tau_{\mathrm{supp}} = 0.05$**（$\hat S_g$ 同理）。最后严格在 $\hat S_c$（与 $\hat S_g$）上求解一次**无惩罚（$\lambda_1=0$）非负最小二乘（NNLS）**消除 $L_1$ 收缩偏差，输出冻结静态骨架 $(\hat S_c, \hat c, \hat S_g, \hat g)$ 移交 Stage 2。

---

### 3.2 Stage 2：残差开放闭合、冻结规则、零泄漏断言与反陷阱训练规范

1. **精确 Stage 2 方程（基准 1/2 振子/摆阵载体，$5N$ 维）**：
   $$\dot\theta = \omega, \qquad M\dot\omega = F S_p p - \tau_e - D\omega \quad (\text{摆阵加已知 } -G\sin\theta),$$
   $$\tau_e = B\big[(c + \beta\odot\tanh(s))\odot\sin\varphi - c\odot\sin\eta\big] + |B|\big[g\odot(\cos\varphi-\cos\eta)\big] + S_e e + \alpha_1\odot\tanh(a) + \alpha_2\odot\tanh^2(a),$$
   - **严禁 `env_edges`**：相对旧 `src/kiki/models/osgkd.py` 第 230 行，**彻底删除**环境端口上的任意无约束边特征直通项 `env_edges`，硬性落实 $\mathcal{F}_{\mathrm{env}}$。
   - **受控残差隐态方程（严禁冻结 $\dot p$）**：
     $$\dot p_i = -\lambda_{p,i} p_i + w_{p\omega,i}\tilde\omega_i + w_{pt,i}\tanh(p_i) + w_{p\omega t,i}\tanh(\tilde\omega_i),$$
     $$\dot a_i = -\lambda_{a,i} a_i + w_{a\omega,i}\tilde\omega_i + w_{at,i}\tanh(a_i) + \sum_{e\in\hat S_c(i)}\big(v_{a,e,i}^{(1)}(\sin\varphi_e-\sin\eta_e) + v_{a,e,i}^{(2)}(\cos\varphi_e-\cos\eta_e)\big),$$
     $$\dot e_i = -\lambda_{e,i} e_i + w_{ea,i} a_i + w_{e\omega,i}\tilde\omega_i + w_{et,i}\tanh(e_i) + w_{eat,i}\tanh(a_i) + \sum_{e\in\hat S_c(i)}\big(v_{e,e,i}^{(1)}(\sin\varphi_e-\sin\eta_e) + v_{e,e,i}^{(2)}(\cos\varphi_e-\cos\eta_e)\big),$$
     所有衰减率均通过 $\lambda_{\cdot,i} = \operatorname{softplus}(\tilde\lambda_{\cdot,i}) + 0.05\ \mathrm{s}^{-1}$ 参数化，保证严格真因果记忆（$T_e^{-1}\ge 0.05$）并从结构上杜绝 $\dot p=0$ 退化；且 $\dot a,\dot e$ 的邻边输入严格掩码至冻结骨架 $\hat S_c$。

2. **骨架冻结规则与唯一投影盒消融**：
   - **主方案（`PDD-Frozen`）**：`c.requires_grad_(False)`、`g.requires_grad_(False)`，所有 $e\notin\hat S_c$ 强制 $c_e=\beta_e\equiv 0$；对 $e\in\hat S_c$，每步更新后硬投影 $0 \le \beta_e \le 0.8\,\hat c_e$。
   - **唯一预登记消融（`Stage-2 Bounded-Recal`）**：支撑 $\hat S_c$ 仍严格冻结（禁增删边），仅允许 $c_e\ (e\in\hat S_c)$ 在定理推导盒内微调：由调制均值偏置 $|\mu_e|\le 0.35$ 及 $\kappa=0.8$ 得 $c_e^{\mathrm{eff}}\in[0.72 c_e, 1.28 c_e]$，加有限样本估计余量给出精确投影盒：
     $$c_e \in \big[0.75\,\hat c_e,\; 1.40\,\hat c_e\big] \quad (\forall e\in\hat S_c), \qquad \text{随即投影 } 0 \le \beta_e \le 0.8\,c_e.$$

3. **等变图短历史编码器 $I_\phi$ 与硬性零 $\mathcal{O}_1$ 泄漏白名单断言**：
   - 架构沿用已验证的 `OSGKDGraph._GraphEncoder`（`src/kiki/models/osgkd_graph.py`，`hidden_size=20`，节点参数共享、参考历史扣减零点对齐，严格保留末点 $(\theta_{\mathrm{obs}}(0),\omega_{\mathrm{obs}}(0))$，**精确 1,928 可训练参数**，保证与 v2 和 `GraphNeuralODE` 编码器容量零偏差）。
   - **静态 + 运行时零泄漏白名单断言（Run 0 强制门禁）**：
     1. 接口层强制 `initialize(packet)` 仅接受 `MotionPacket`，校验顶层键集合**精确等于** `{"history", "constants"}`，且 `history` 键严格限定为 `{"t", "theta", "omega"}`（基准 3 为 `{"t", "r"}`），出现任何特权键（`targets`, `tau_m`, `tau_e`, `I_drive`, `I_net`, `truth`, `label`, `p`, `a`, `e`, `u`, `m`）立即抛 `PermissionError`；
     2. 静态 AST 检查验证 `initialize()` 与 `rollout()` 源码及调用链零引用特权字段名；
     3. 运行时污染单测（Taint Test）：将验证事件的全部 $\mathcal{O}_1$ 字段覆写为 `NaN` 与 `+1e9` 两份副本，断言 `initialize` 与 `rollout` 输出与原事件**逐比特完全一致（`atol=0, rtol=0`）**。

4. **损失权重、课程与优化器（显式规避 `frozen_p` 3× 收缩陷阱与 `P2a` 历史精修灾难）**：
   - **总目标**（$\mathcal{O}_1$ 仅作训练监督目标，绝不进入输入）：
     $$\mathcal{L}_{\mathrm{Stage2}} = \mathcal{L}_{\mathrm{motion}} + 0.5\,\mathcal{L}_{\mathrm{port}} + 0.5\,\mathcal{L}_{\mathrm{port},0} + 0.05\,\mathcal{L}_{\mathrm{cons}} + 10^{-4}\|\beta\|_1,$$
     - $\mathcal{L}_{\mathrm{motion}}$：归一化 $(\theta,\omega)$ 在有效未来区间 $(0, T_s]$ 的均方误差（失稳轨线在截止点截断，不填充）；
     - $\mathcal{L}_{\mathrm{port}}$：归一化 $(\tau_m, \tau_e)$ 在 $(0, T_s]$ 的均方误差（权重 `0.5`，强于 v2 的 `0.1`，锚定开放残差归属）；
     - **$\mathcal{L}_{\mathrm{port},0}$（反 `frozen_p` 3× 收缩专设项）**：在部署起点 $t=0$ 直接以 $\mathcal{O}_1$ 监督 $\hat\tau_m(0)=F S_p p_0$ 与 $\hat\tau_e(0)$（权重 `0.5`），配合可训练阻尼 $\dot p$，彻底杜绝编码器压缩 $\|p_0\|_2$ 的跨模块代偿；
     - $\mathcal{L}_{\mathrm{cons}}$：推进 $0.1\ \mathrm{s}$ 后隐态与编码器对未来 $0.1\ \mathrm{s}$ 运动窗重新编码的一致性损失（权重 `0.05`）；
     - **永久禁用 `P2a` 历史一步精修**：无论训练期或部署期均不做初态梯度精修，杜绝过拟合 11 帧噪声导致未来风险恶化 44× 的灾难。
   - **三段课程与优化器**：Phase A（$T_s=1\ \mathrm{s}$, 64 步）$\to$ Phase B（$T_s=2\ \mathrm{s}$, 64 步）$\to$ Phase C（$T_s=4\ \mathrm{s}$, 128 步），总计 **`256` 次接受更新**；物理与有效批量均为 `4`（按激励/稳失稳分层配平）；Adam 分组学习率（编码器 `3e-3`、残差 ODE `1e-2`），8 步线性预热 + 余弦退火至 `0.1×`，梯度范数裁剪 `5.0`，每步后硬投影约束与支持掩码；训练内部 RK4 步长 `0.01 s`（训练前强制过 `0.01 vs 0.005 s` 双网格梯度资格），评分强制过 `0.005 vs 0.0025 s` 双网格资格。

---

## 四、基准 1 强开放/强调制工况定量设计、十字星矩阵、ML 基线公平口径与 G1–G4 硬门禁

### 4.1 旧 Layer 1 诊断：为何 `Stage-1-Only` 拿到 `0.00207` 风险？

对照逐行审查的 `src/kiki/data/osgkd_synthetic.py`（L57–113）与 `experiments/osgkd_v2_wp2_layer1_20261127/models.py`（L22–45），旧 Layer 1 中 `weak_static` 在 `modulated` 无噪格仅 `0.00207`、在 `open` 无噪格仅 `0.00705` 的**五重定量根因**完全明确：

1. **调制能量微弱（$<4\%$ RMS 占比）**：仅 $4/12$ 条边带调制且 $\beta_e = 0.3 c_e$（L60–61）；初态 $a_0\sim\mathcal{N}(0,0.4^2)$ 又按 $\dot a=-0.6a+0.3\omega$ 迅速衰减（L85, L104），导致调制转矩在 4 s 内的 RMS 不到静态耦合转矩的 $4\%$。
2. **开放环境记忆弱且纯线性衰减**：`environment_gain = 0.4`（L65），而节点静态耦合刚度和约 $3.0$（大 $7.5\times$）；$\dot e=-0.4e+0.2\omega$（L86）无非线性/相角反馈，环境项在 4 s 内单调衰减，对运动总方差贡献极低。
3. **小摆角线性域、零失稳事件**：初态扰动仅 $\theta_0\sim\mathcal{N}(0,0.2^2), \omega_0\sim\mathcal{N}(0,0.12^2), p_0\sim\mathcal{N}(0,0.2^2)$（L104），最大角差 $|\varphi_e|\lesssim 0.4\ \mathrm{rad}\ll\pi$，**0/48 条失稳**，几乎无大角非线性效应。
4. **`WeakStatic` 仍配齐了 1,928 参编码器与线性机械通道吸纳残差**：`WeakStatic`（`models.py` L22–43）虽冻结 $\hat c$，但仍联合训练完整图编码器及 $\dot p=a_p p+b_p\omega$；在小扰动线性域内，线性机械态 $p(t)$ 足以吸收微弱线性记忆 $0.4e(t)$ 的大部分动量效应。
5. **旧弱式回归用正值低通 Hann 窗却未被惩罚**：因旧环境记忆弱，低通正窗虽然把 $S_e e(t)$ 全部透传，仍未破坏 $\hat c$ 的支撑恢复（相对误差仅 `0.048`）。

### 4.2 新基准 1（`open_oscillator_v3`）强开放/强调制与大样本失稳定量参数设计

| 物理组件 | 旧 Layer 1（过弱） | **新基准 1 `open_oscillator_v3`（唯一拍板规格）** | 物理与统计作用 |
| --- | --- | --- | --- |
| **稀疏拓扑与参考** | $N=8$（12/28 边），16/32 为克隆拷贝；$\theta_\star=0$ | $N\in\{8,16,32\}$ **原生随机稀疏图**（环+$\lfloor N/2\rfloor$ 弦，$|S^\star|=1.5N$：$12/28$、$24/120$、$48/496$）；非零稳态流 $\theta_{\star,i}\sim\mathcal{U}(-0.35,0.35)$ 居中 | 考验真实图稀疏恢复与一般 $\eta\ne 0$ 下的 Theorem 1 |
| **惯量、阻尼与静态骨架** | $M\in[0.8,1.2], D\in[0.15,0.30], c\in[0.7,1.3]$ | $M_i\sim\mathcal{U}(0.7,1.4)$，$D_i\sim\mathcal{U}(0.12,0.25)$，$c_e\sim\mathcal{U}(1.2,2.4)$ | 轻阻尼维持 4 s 宽带多频振荡，保障 (A3) 强制性 |
| **强动态调制（奇对称核）** | $1/3$ 边，$\beta=0.3c$，无角反馈（RMS $<4\%$） | **$2/3$ 活动边**带调制，$\beta_e\sim\mathcal{U}(0.65,0.80)\,c_e$；核 $K_e(a)=c_e+\beta_e\tanh(a_i+a_j)$（刚度可在 $[0.2c_e,1.8c_e]$ 变动 9×）；$\dot a_i=-\lambda_{a,i}a_i+\gamma_{a\omega,i}\omega_i+\gamma_{a,\cos,i}\sum_{e\in S^\star(i)}(\cos\varphi_e-\cos\eta_e)$，$\lambda_a\in[0.35,0.55], \gamma_{a\omega}\in[0.45,0.70], \gamma_{a,\cos}\in[0.30,0.50]$ | **大角正反馈“电压跌落”机制**：$|\varphi_e|$ 增大 $\Rightarrow \cos\varphi_e-\cos\eta_e<0 \Rightarrow a_i$ 变负 $\Rightarrow K_e(a)$ 骤降至 $0.2c_e$，使静态 `Stage-1-Only` 严重高估同步回复力而漏报失稳！**生成器断言调制转矩 RMS 占 $\tau_e$ 的 `22%–35%`**。 |
| **强开放环境记忆（$\mathcal{F}_{\mathrm{env}}$）** | $S_e=0.4$，$\dot e=-0.4e+0.2\omega$（无角/幅反馈） | $\tau_{\mathrm{env}}^{\mathrm{open}}=S_e e + \alpha_{\mathrm{cm}}\odot\tanh(a)$，$S_{e,i}\in[1.4,2.2], \alpha_{\mathrm{cm},i}\in[0.25,0.45]$；$\dot e_i=-T_{e,i}^{-1}e_i + \gamma_{e\omega,i}\omega_i + \gamma_{ea,i}\tanh(a_i) + \gamma_{e,\cos,i}\sum_{e\in S^\star(i)}(\cos\varphi_e-\cos\eta_e)$，$T_e^{-1}\in[0.18,0.35]\ \mathrm{s}^{-1}, \gamma_{e\omega}\in[0.35,0.60], \gamma_{ea}\in[0.25,0.45], \gamma_{e,\cos}\in[0.15,0.30]$ | 慢时标（$T_e\in[2.85,5.55]\ \mathrm{s}$）与 Stage 1 通带分离 $>9\times$；强非线性开放功率交换使 `Stage-1-Only` 长时严重失配（目标 4 s MSE $\ge 0.10$）。**生成器断言环境转矩 RMS 占 $\tau_e$ 的 `35%–50%`**。 |
| **对称余弦损耗切片** | 无 | `strong_open_sym` 格加入 $|B|[g\odot(\cos\varphi-\cos\eta)]$，$g_e\sim\mathcal{U}(0.25,0.50)c_e$（$e\in S^\star$） | 在非二分图上实证检验 Theorem 1 机制 1 |
| **主动机械驱动** | $\dot p=-0.8p-0.2\omega$ | $\dot p_i=-\lambda_{p,i}p_i-\gamma_{p\omega,i}\omega_i-\gamma_{pt,i}\tanh(p_i)$，$\lambda_p\in[0.40,0.65], \gamma_{p\omega}\in[0.25,0.45], \gamma_{pt}\in[0.10,0.20]$ | 非平凡动态调速，防止单靠常数 $p_0$ 拟合 |
| **大样本失稳与防作弊分界生成** | 48 train / 24 val，**0 失稳** | 每格固定 **`train = 96`（64 稳 + 32 失稳）、`val = 32`（16 稳 + 16 失稳）、`test = 128`（64 稳 + 64 可初始化失稳）**。失稳判据：$(0,4]\ \mathrm{s}$ 内连续 5 帧（$0.05\ \mathrm{s}$）满足 $\operatorname{ptp}(\theta)>\pi$ 且 $\max_i|\omega_i-\omega_{\mathrm{COI}}|>0.35\ \mathrm{rad/s}$；历史窗 $[-0.1,0]\ \mathrm{s}$ 内强制非失步（**100% 因果可初始化**）。4 类分层激励：`distributed_strong`、`local_fault`、`latent_overload`（强 $(p_0,a_0,e_0)$ 而初角/初速温和，防简单阈值作弊）、`boundary_near_separatrix`（沿随机扰动射线 1D 二分包夹分界 $\pm 15\%$ 生成成对稳/失稳样本，保证稳/失稳初始能量分布显著重叠）。 | 64 失稳 + 64 稳测试集使召回 95% CI 收窄至 $\pm 10\%$，配对 McNemar 对 20 pp 差异检验力 $>95\%$，DeLong AUROC 标准误 $\approx 0.025$。 |

### 4.3 十字星实验矩阵（严格控算力，禁全笛卡尔积）

| 切片轴 | 格点设置 | 方法清单 | 种子与独立单元 |
| --- | --- | --- | --- |
| **中心锚格（Anchor）** | 基准 1，$N=8$，`strong_open`，历史噪声 `0.01` | **全部 13 端点**：`PDD-Frozen`、`PDD Bounded-Recal`、`Stage-1-Only`、`End-to-End OSGKD v2`、`No-Privilege Same-Arch`、`NRI/ACD`、`NRI+Port`、`SINDy-AE`、`Latent ODE`、`PHNN/LGNN`、`Equivariant Graph Neural ODE`、`Direct Classifier`、`Seq2Seq` | **3 种子**（`24121, 24122, 24123`）；另对种子 1 的 `No-Privilege` 追加 384 步充分性检查 |
| **纵轴（工况递进）** | $N=8$，噪声 `0.01`：① `static_closed`；② `strong_mod_closed`；③ `strong_open`（复用锚格）；④ `strong_open_sym` | 核心 5 方法（`PDD-Frozen`, `Stage-1-Only`, `End-to-End v2`, `No-Privilege`, `Graph Neural ODE`）；格 ②④ 追加 `PDD Bounded-Recal` | 格 ①②：**2 种子**；格 ③④：**3 种子** |
| **横轴 A（噪声）** | $N=8$，`strong_open`，噪声 `{0.00, 0.01(复用), 0.03}` | 核心 5 方法 | 格 `0.00, 0.03`：**2 种子** |
| **横轴 B（规模）** | `strong_open`，噪声 `0.01`，$N\in\{8(\text{复用}), 16, 32\}$ | 核心 4 方法（`PDD-Frozen`, `Stage-1-Only`, `End-to-End v2`, `Graph Neural ODE`） | $N=16$：**2 种子**；$N=32$：**1 种子**（先过 Run 0 内存 dry-run） |
| **基准 2（升级摆阵）** | $N=8$，强开放/强调制 + 已知重力 $-G\sin\theta$，每系统 `96/32/128`（含 64 翻转/失稳），噪声 `0.01` | 核心 6 方法（`PDD-Frozen`, `Stage-1-Only`, `End-to-End v2`, `No-Privilege`, `Graph Neural ODE`, `PHNN/LGNN`） | **3 个独立系统种子**（`31, 41, 51`）各 1 优化种子（3 独立物理系统） |
| **基准 3（Wilson–Cowan）** | $N=8$（6E+2I，强 STP+适应记忆，64 稳+64 癫痫）及 $N=16$（12E+4I），噪声 `0.01` | $N=8$：**10 个适用端点**（`PDD-Frozen`, `Stage-1-Only`, `End-to-End v2-WC`, `No-Privilege`, `NRI/ACD`, `SINDy-AE`, `Latent ODE`, `Graph Neural ODE`, `Direct Classifier`, `Seq2Seq`；非哈密顿故 `PHNN/LGNN` 标注 N/A）；$N=16$：核心 4 方法 | $N=8$：**3 种子**；$N=16$：**1 种子** |
| **理论数值核验** | Theorem 2 四类负对照 + 20 格激励/重合相图 + Theorem 3 逃逸对照（$e^z$ vs $1+0.8\tanh z$ vs $e^{\tanh z}$） | 确定性 NumPy 验证脚本 | 1 次确定性运行（$<2\ \mathrm{min}$） |

### 4.4 全部 10 个必配基线的公平口径表（信息合同、容量匹配、优化预算）

> **统一公平原则**：所有动力学基线均在 CPU float64 下以相同 3 段课程（`64@1s + 64@2s + 128@4s = 256` 步、批量 `4`、RK4 $\Delta t_{\mathrm{train}}=0.01\ \mathrm{s}$、相同 Adam+余弦学习率）训练，并在同一 `0.005/0.0025 s` 双网格资格与同一 128 条冻结测试集上评分；$N=8$ 可训练参数量统一匹配在 **`~2,000–3,000`** 区间（消除参数量悬殊混淆）；非 ODE 参照（`Direct Classifier`, `Seq2Seq`）给予充足收敛预算（早停于 `val`）作为诚实参照上界。

| # | 方法名称 | 训练期信息合同 | 部署期输入 | $N=8$ 参数量（估计） | 关键实现与公平性边界 |
| :-: | --- | :---: | :---: | ---: | --- |
| 主 | **`PDD-Frozen`**（主方案） | $\mathcal{O}_1$（Stage 1 弱式 + Stage 2 监督） | 仅 $\mathcal{O}_0$ | `~2,300` | Stage 1 冻结 $(\hat S,\hat c,\hat g)$；Stage 2 仅学 $\beta_{\hat S}$、$\mathcal{F}_{\mathrm{env}}$ 残差与 1,928 参图编码器（`Bounded-Recal` 同参数，仅开放 $c_{\hat S}\in[0.75\hat c,1.40\hat c]$） |
| 1 | **`Stage-1-Only`** | $\mathcal{O}_1$（仅 Stage 1 + 简单线性 $\dot p$） | 仅 $\mathcal{O}_0$ | `~1,972` | 核心消融（直面 Weak SINDy 质疑）：精确相同 Stage 1 冻结 $\hat c$，关闭 $\beta\equiv 0$、$e\equiv 0$，仅训线性 $\dot p$ 与 $p_0$ 编码器（256 步） |
| 2 | **`End-to-End OSGKD v2`** | $\mathcal{O}_1$（全量联合） | 仅 $\mathcal{O}_0$ | `2,376` | 严格保留本仓冻结 `OSGKDGraph`（指数核、含 `env_edges`、128 步密训 + 128 步剪枝重拟合 = 256 步） |
| 3 | **`No-Privilege Same-Arch`** | **仅 $\mathcal{O}_0$ 轨迹**（零端口监督） | 仅 $\mathcal{O}_0$ | `~2,320` | 与 PDD 完全同构（奇对称核、无 `env_edges`、同 1,928 参编码器），跳过 Stage 1，仅靠未来 $(\theta,\omega)$ 运动损失 + $L_1(c,\beta)$ 训练 256 步（锚格种子 1 另跑 384 步排除欠优化） |
| 4 | **`NRI / ACD`** | 标配 **$\mathcal{O}_0$ 轨迹**（锚格另列 `NRI+Port` 含 $\mathcal{O}_1$ 读出头） | 仅 $\mathcal{O}_0$ | `~2,800` | Kipf+ (ICML'18) / Löwe+ (CLeaR'22) 摊销交互图推断：GNN 边编码器经 Gumbel-Softmax 推断二值边概率 $q_\phi(z_{ij}\mid H_{100})$，配匹配 RK4 GNN 解码器（256 步） |
| 5 | **`SINDy-AE`** | $\mathcal{O}_1$（运动 + 端口监督） | 仅 $\mathcal{O}_0$ | `~2,500` | Champion+ (PNAS'19) 部分观测稀疏发现：共享 1,928 参编码器 + 候选三角/多项式/$\tanh$ 稀疏字典，每 32 步执行一次序贯阈值剪枝（STLSQ），共 256 步 |
| 6 | **`Latent ODE`** | $\mathcal{O}_1$（运动 + 端口解码监督） | 仅 $\mathcal{O}_0$ | `~2,600` | Rubanova+ (NeurIPS'19) 黑箱隐态连续动力学：GRU 编码 $H_{100}\to h_0\in\mathbb{R}^{5N}$，无图互易先验 MLP ODE 演化 $h(t)$ 并线性解码 $(\theta,\omega,\tau_m,\tau_e)$（256 步） |
| 7 | **`PHNN / LGNN`** | $\mathcal{O}_1$（运动 + 端口监督） | 仅 $\mathcal{O}_0$ | `~2,500` | Desai+ (PRE'21) / Thangamuthu+ (NeurIPS'22)：共享 1,928 参编码器 + 图势能哈密顿网络 $H_\theta(\theta,\omega)$ + 半正定耗散 $R_\theta\succeq 0$ + 外端口头（仅适用基准 1、2；基准 3 标 N/A） |
| 8 | **`Equivariant Graph Neural ODE`** | $\mathcal{O}_1$（运动 + 端口监督） | 仅 $\mathcal{O}_0$ | `2,599` | 严格保留本仓 `GraphNeuralODE`（同 1,928 参等变图编码器 + 稠密边/节点 MLP RHS，无互易与 $\mathcal{F}_{\mathrm{env}}$ 结构约束，256 步） |
| 9 | **`Direct Classifier`** | $\mathcal{O}_0$ 历史 + **二元失稳标签** | 仅 $\mathcal{O}_0$ | `~2,150` | **非动力学分类参照上界（显式标注不产出轨迹、端口或图）**：同 1,928 参图 GRU 编码器 + 池化二分类头，BCE 训至 `val` 早停收敛（$\le 200$ epoch） |
| 10 | **`Seq2Seq`** | $\mathcal{O}_0$ 历史 + 未来 $(\theta,\omega)$ | 仅 $\mathcal{O}_0$ | `~2,800` | **非 ODE 序列回归参照（显式标注无自治微分方程与拓扑）**：同图 GRU 编码器 + 多步序列解码器直出 400 帧未来运动，MSE 训至 `val` 早停收敛 |

*结构恢复评测边界*：仅显式边模型（`PDD`, `Bounded-Recal`, `Stage-1-Only`, `End-to-End v2`, `No-Privilege`, `NRI/ACD`, `SINDy-AE`）计入拓扑 AUROC/AUPRC/F1 与 $\mathcal{O}_2$ 端口分离误差；黑箱/隐式模型明确标为 `—`，不伪造结构指标。

### 4.5 G1–G4 预登记硬门禁、统计检验方法与止损触发条件

所有硬门禁在**中心锚格 3 种子、128 条冻结测试集（64 稳 + 64 失稳）**上按以下统一指标与检验判定（`VPT` 定义为归一化瞬时状态误差 $\|\hat x(t)-x(t)\|_2/\mathrm{RMS}(x)$ 首次超过 $0.20$ 前的最大有效预测时长，上限 $4.0\ \mathrm{s}$；失稳 AUROC 采用自治轨线最大摆幅 $s_{\mathrm{instab}}=\max_{t\in[0,4]}\operatorname{ptp}(\hat\theta(t))$，无额外分类头）：

| 门禁 | 核心比较对 | 预登记定量通过阈值（须**全部同时满足**） | 统计检验与区间方法 | 失败 / 止损触发条件 |
| --- | --- | --- | --- | --- |
| **G1（Stage 2 必要性）** | `PDD-Frozen` vs `Stage-1-Only` | ① 3 种子均值 `MSE@4s` 相对降低 **$\ge 40\%$**（比值 $\le 0.60$），且 **3/3 种子同号**；<br>② 3 种子均值 `VPT` 净提升 **$\ge +0.60\ \mathrm{s}$**（3/3 种子同号）；<br>③ 失稳召回净提升 **$\ge +15\ \mathrm{pp}$**（同时稳定误报率 $\le 10\%$）**或** 失稳 `AUROC` 提升 **$\ge +0.08$**。 | 逐事件配对自助法（Paired Event Bootstrap, $B=10{,}000$）计算 95% CI（严格不跨零）；召回采用 **McNemar 精确配对检验**（$p<0.01$）；AUROC 采用 **DeLong 配对检验**（$p<0.01$）。 | **Run 0 早停止损**：若 `Stage-1-Only` 的 `MSE@4s < 0.05` 或 PDD 相对降幅 $<25\%$，暂停全矩阵，先核查生成器调制/开放 RMS 占比是否落入目标区间及 Stage 2 是否欠拟合。 |
| **G2（解耦优于端到端）** | `PDD-Frozen` vs `End-to-End OSGKD v2` | ① 静态边 `AUPRC` 提升 **$\ge +0.15$**，静态边 `F1 >= 0.90` 且高出 v2 **$\ge +0.15$**（3/3 种子）；<br>② $\mathcal{O}_2$ 内部/环境分离 `Port Sep RMSE` 相对降低 **$\ge 30\%$**（3/3 种子）；<br>③ 预测/失稳非退化护栏：`MSE@4s` 不劣于 v2 超过 $10\%$ 且失稳 `AUROC` 不劣于 v2 超过 $0.03$。 | 配对 Bootstrap 95% CI（不跨零）；`Port Sep RMSE` 配对 **Wilcoxon 符号秩检验**（$p<0.01$）。并同步核查 `Bounded-Recal`：仅当其跨 3/3 种子降 `MSE@4s` $\ge 10\%$ 且 $\hat c$ 相对误差恶化 $\le 5\%$ 时才记为有效补偿消融，否则坚持 `PDD-Frozen` 主干。 | 若 v2 在长时预测与失稳上全面碾压（PDD 退化 $>15\%$）或静态边 F1 优势 $<0.05$，放弃硬冻结解耦主干主张，转查 Stage 1 残差偏差传递。 |
| **G3（特权信息纯增益）** | `PDD-Frozen` vs `No-Privilege Same-Arch` | ① 静态边 `AUPRC` 提升 **$\ge +0.15$** 且 `F1` 提升 **$\ge +0.15$**（3/3 种子）；<br>② $\mathcal{O}_2$ `Port Sep RMSE` 相对降低 **$\ge 35\%$**（3/3 种子）；<br>③ `MSE@4s` 相对降低 **$\ge 15\%$** 或失稳 `AUROC` 提升 **$\ge +0.05$**（另一项退化 $\le 5\%$）。 | 配对 Bootstrap 95% CI；并在锚格种子 1 对 `No-Privilege` 跑 $1.5\times$（384 步）充分性校验以排除“无特权仅收敛慢”的混淆。 | 若无特权同构在 384 步追平 PDD（F1 差 $<0.05$ 且 `Port Sep` 差 $<10\%$），则 LUPI 头条不成立，触发 Campaign 结构复盘。 |
| **G4（定性失稳分岔保真）** | `PDD-Frozen` vs `Graph Neural ODE` & `Latent ODE`（并列 `Direct Classifier`） | ① `PDD-Frozen` 失稳召回 **$\ge 70\%$**、稳定误报 **$\le 12\%$**，且召回高于两黑箱 ODE 至少 **$\ge +20\ \mathrm{pp}$**；<br>② 失稳 `AUROC >= 0.85` 且高于两黑箱 ODE **$\ge +0.08$**；<br>③ 稳定子集 `MSE@4s` 不超 `Graph Neural ODE` 的 $1.5\times$；<br>④ 诚实并列 `Direct Classifier`（不强求反超有监督分类器，展示在产出连续轨迹+物理图前提下的可比性）。 | 64 条失稳 **McNemar 精确检验**（$p<0.005$）；128 条 **DeLong 检验**（$p<0.005$）；精确 **Clopper–Pearson 95% CI**。并在基准 3（Wilson–Cowan 64+64 癫痫跃迁）复验跨方程族一致性。 | 若黑箱 ODE 在大样本合成失稳上亦保持高召回（$>65\%$ 且与 PDD 差 $<10\ \mathrm{pp}$），按证据立即收缩 C3/G4 主张至“高阶约化失配/强分布外（如 IEEE-39）情形”，不夸大。 |

---

## 五、单进程 CPU float64 算力预算与 Run 0 最小判别计划

### 5.1 基于 `resource_budget.md` 实测的单进程 CPU float64 分阶段算力预算

直接锚定 `reports/osgkd_v2_wp2_layer1_20261127/resource_budget.md` 实测数据（单进程 CPU float64，批量 4，RK4 $\Delta t=0.01\ \mathrm{s}$）：
- $N=8$（28 候选边）：$2\ \mathrm{s}$ `1.028 s/step`（RSS `715 MiB`），$4\ \mathrm{s}$ `1.837 s/step`（RSS `852 MiB`），插值 $1\ \mathrm{s}$ `~0.58 s/step`；单次 256 步三段课程（`64@1s + 64@2s + 128@4s`）裸时长 `338 s (5.6 min)`，乘以 **1.4× 工程保守系数**（含双网格资格、检查点 I/O、`GraphNeuralODE` 稠密边 MLP 及后台负载）按 **`0.132 h/run`**（`GraphNeuralODE` 按 `0.18 h/run`，`Stage-1-Only` 按 `0.06 h/run`，`Direct Classifier`/`Seq2Seq` 按 `0.03 h/run`）计；128 条测试集批量 16 双网格（`0.005/0.0025 s`）评分约 `0.75 min/endpoint`。
- $N=16$（120 候选边）：$2\ \mathrm{s}$ `0.987 s/step`（`780 MiB`），$4\ \mathrm{s}$ `2.144 s/step`（`982 MiB`）；保守 **`0.15 h/run`**（`GraphNeuralODE` 约 `0.30 h/run`）。
- $N=32$（496 候选边）：$2\ \mathrm{s}$ `1.914 s/step`（`1,135 MiB`），$4\ \mathrm{s}$ `3.588 s/step`（`1,685 MiB`）；保守 **`0.25 h/run`**（`GraphNeuralODE` $N^2=1024$ 稠密消息约 `0.63 h/run`，峰值 RSS 预计 `2.2–2.8 GiB`，须先过 Run 0 内存 dry-run）。
- 基准 3（Wilson–Cowan，$4N$ 态）：约为同 $N$ 振子单步耗时的 **0.8×**（$N=8$ 约 `0.11 h/run`，$N=16$ 约 `0.13 h/run`）。

| 执行阶段 | 作业规模（运行条数 × 课程） | 总梯度更新次数 | 净单进程 CPU 墙钟（含 1.4× 系数与双网格评分） | 预计峰值工作集（RSS） |
| --- | --- | ---: | ---: | ---: |
| **Run 0（资格 + G1/G2 冒烟 + 内存 Dry-Run）** | 基准 1 锚格 1 种子 × 5 端点 + 基准 3 冒烟 1 种子 × 2 端点 + 零泄漏/理论/内存 dry-run | **`1,792`** | **`1.4 h`** | `0.9 GiB`（$N=32$ dry-run $\le 2.8\ \mathrm{GiB}$） |
| **Phase 1 · 基准 1 中心锚格** | 11 ODE 端点 × 3 种子 + 1×384 步充分性校验 + 6 非 ODE 快跑 | `8,576` | **`5.4 h`** | `0.95 GiB` |
| **Phase 1 · 基准 1 纵轴切片** | `static_closed`(10) + `strong_mod_closed`(12) + `strong_open_sym`(18) = 40 runs | `10,240` | **`5.7 h`** | `0.95 GiB` |
| **Phase 1 · 基准 1 横轴 A（噪声）** | `0.00 / 0.03` × 2 种子 × 5 方法 = 20 runs | `5,120` | **`3.0 h`** | `0.95 GiB` |
| **Phase 1 · 基准 1 横轴 B（规模）** | $N=16$ (8 runs) + $N=32$ (4 runs) | `3,072` | **`3.4 h`** | `1.0–2.8 GiB` |
| **Phase 2 · 基准 2 升级机械摆阵** | 3 独立系统 × 6 核心方法 = 18 runs | `4,608` | **`2.8 h`** | `0.95 GiB` |
| **Phase 3 · 基准 3 Wilson–Cowan** | $N=8$ (8 ODE × 3 种子 + 6 非 ODE) + $N=16$ (4 核心 × 1 种子) | `7,168` | **`3.8 h`** | `0.85–1.2 GiB` |
| **核心全套合计（零新 FOM、零 `test`）** | **128 ODE runs + 12 非 ODE runs + 理论核验** | **`40,576`** | **`25.5 h`（约 1.06 天）** | **$\le 2.8\ \mathrm{GiB}$** |
| **含 2.5× 诊断/调优/复跑安全余量** | 预留 Run 0 回调、超参核验、失败重跑与图表复算 | `~100,000` | **`~64 h`（< 2.7 天单进程 CPU）** | **$\le 2.8\ \mathrm{GiB}$** |
| *可选 Phase 4 · IEEE-39 `train/val` 探索对比* | *仅在基准 1–3 闭合且另行登记后决定（严格零 `test` 访问）* | *`~1,536`* | *`~6–8 h`* | *`~1.2 GiB`* |

**NeurIPS 2027（约 5 个月 / 20 周）日程可行性核验**：
全套核心实验（含 2.5× 安全余量）的单进程 CPU 纯计算墙钟不足 **3 天**，仅占 5 个月周期的 **< 2%**，峰值内存严格低于 `3 GiB`，单机单进程即可极其从容地完成全部实验，从而将 95% 以上研发周期聚焦于**定理证明严密化、零泄漏与公平基线工程审查、Run 0 早停把关及高质量论文撰写与模拟审稿**。

---

### 5.2 放行后 Run 0（资格 + G1 冒烟探针）最小判别计划

- **触发前提**：待本暂定设计经 `root` 审阅、独立 Reviewer 只读挑战并获用户正式放行代码实现与 Run 0 后启动；**当前保持 `PAUSED_BY_USER` 与 `default none`**。
- **最小工作量（总墙钟 $\le 1.5\ \mathrm{h}$，`1,792` 次更新）**：
  1. **零训练硬资格包（约 `10 min`）**：① 静态 AST + 运行时 `NaN/1e9` 污染零 $\mathcal{O}_1$ 泄漏断言；② 基准 1 与基准 3 生成器物理强度断言（验证基准 1 调制 RMS $\in [22\%,35\%]$、环境记忆 RMS $\in [35\%,50\%]$、稳/失稳初始能量分布重叠、100% 因果可初始化；验证基准 3 稳态/癫痫态分离与正向不变性）；③ Theorem 2 四类负对照 + Theorem 3 双机逃逸（$e^z$ vs $1+0.8\tanh z$ vs $e^{\tanh z}$）自动化复核；④ $N\in\{8,16,32\}$（含 `GraphNeuralODE`）4 s 批量 4 单进程内存与耗时 dry-run。
  2. **基准 1 锚格冒烟（$N=8$, `strong_open`, 噪声 `0.01`, 单种子 `24121`, 5 端点 = `1,280` 步, 约 `50 min`）**：`Stage-1-Only`、`PDD-Frozen`、`PDD Bounded-Recal`、`End-to-End OSGKD v2`、`Equivariant Graph Neural ODE`，统一在 128 条冻结测试集（64 稳 + 64 失稳）上过双网格评分。
  3. **基准 3 冒烟（Wilson–Cowan $N=8$, 单种子 `24121`, 2 端点 = `512` 步, 约 `18 min`）**：`Stage-1-Only` vs `PDD-Frozen`，核验非振子有向突触恢复与双稳态癫痫跃迁判别。
- **Run 0 明确通过（Go）判据**：
  1. 全部 4 项零训练资格 100% 通过（0 泄漏、两网格合格率 100%、$N=32$ 峰值 RSS $\le 4.0\ \mathrm{GiB}$）；
  2. 基准 1 锚格（种子 `24121`）：Stage 1 静态边 `F1 >= 0.90`（`AUPRC >= 0.95`）；`Stage-1-Only` 出现真实长时/失稳失配（`MSE@4s >= 0.10` 或失稳召回 $\le 65\%$）；`PDD-Frozen` 较 `Stage-1-Only` 的 `MSE@4s` 降幅 $\ge 35\%$ 且 `VPT` 提升 $\ge +0.50\ \mathrm{s}$；较 `End-to-End v2` 的静态边 `F1` 高出 $\ge +0.15$；
  3. 基准 3（种子 `24121`）：Stage 1 有向突触 `AUPRC >= 0.90`，`PDD-Frozen` 较 `Stage-1-Only` 的 `MSE@4s` 降幅 $\ge 30\%$ 且癫痫跃迁召回 $\ge 65\%$。
- **反面结果如何改变决策（四条精确止损/转向分支）**：
  - **分支 A（`Stage-1-Only` `MSE@4s < 0.05`）**：说明新基准开放/调制仍偏弱（重蹈旧 Layer 1 覆辙）；**冻结全矩阵扩张**，仅依据生成器 RMS 诊断上调耦合调制或开放记忆增益并重跑 Run 0，不改算法架构。
  - **分支 B（Stage 1 静态边 `F1 < 0.85`）**：说明强开放记忆或大调制波动穿透通带；检查 $\Phi_{\mathrm{loc}}$ 投影后的带通残差谱与 $\sigma_{\min}(\mathcal{G})$，核实是激励带宽不足（违反 A3）还是需收窄通带下限（如移除 $0.5\ \mathrm{Hz}$ 尺度 2 仅留 $1\text{–}3\ \mathrm{Hz}$ 尺度 1）。
  - **分支 C（`PDD-Frozen` 劣于 `PDD Bounded-Recal` $\ge 15\%$ 或劣于 v2 $\ge 15\%$）**：诊断是否因大摆角电压跌落使 $a(t)$ 呈显著负均值偏置（$\mu_e < -0.25$）导致 Stage 1 系统低估 $c^\star$；若确证为非零均值偏置，则在进入 Phase 1 前将预登记主干升级为 `Stage-2 Bounded-Recal`（仍严格冻结 $\hat S$），并在 Theorem 1 中明示非对称 $a(t)$ 的箱式重校准边界。
  - **分支 D（`Graph Neural ODE` 在大样本合成失稳上召回 $\ge 75\%$）**：不构成算法失败，但立即复核稳/失稳初始能量分布是否过于易分；若在严格重叠分界下黑箱 ODE 仍高召回，则**诚实收缩论文 C3/G4 主张范围**至高阶约化失配情形（IEEE-39），避免过度宣称，聚焦 C1+C2+C4 主线推进。

---

*报告完毕（Provisional Scientist-Role Bounded Design Drafter，严格只读，未修改任何文件或运行任何实验）。*
