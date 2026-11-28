# PDD v3.1（OSGKD-Decoupled）正式 Scientist 有界设计方案

**设计状态：`SCIENTIST_APPROVED_V3_1`；独立 Reviewer 对 v3.1 的复核待完成。执行状态：`PAUSED_BY_USER`；模型指针：`default none`。**

本稿正式合入 [v3_1_scientist_spec.md](v3_1_scientist_spec.md) 的完整规范与静态复算摘录；本次数字门禁通过后由 `gpt-6.1-sol` 仅作文档落地。科学审定作者与回执归属保持下述 `gpt-6-astra` 科学分工，文档实现者不作科学重判，不构成独立复核、训练放行或模型晋升。

**历史来源与作者区别。** v3.0 原稿由 Antigravity 只读 research 子代理、模型 `gemini-4-neon-xhigh` 起草，会话 `1639b919-2a4c-4032-b93d-7768bcefc975`，当时仅为暂定草案。原稿以 Git `HEAD:reports/pdd_v3_architecture_20261128/scientist_design.md` 的冻结内容为准，统一 LF 后 SHA-256 为 `ab98674c2430b5d2569261db8de1faa66c02db5645759a108a7d46c6965fb9df`（UTF-8 字节；CRLF 先转 LF）。此前本机 `codex-cli 0.144.4` 尝试 `gpt-6-astra` 得 HTTP 400（要求升级 Codex）的事实与 [历史回执](../../results/pdd_v3_architecture_20261128/scientist_receipt.json) 不追改，也不代表本次审定失败。原作者的“0 文件修改”等只读声明仅描述原次调用。本次正式文本替换过时矛盾内容，未将原稿作者改记为指定 Scientist。

本文前九节与附录为完整 v3.1 规范；后部保留章节给出原方案中不冲突的 Stage 2、基准 2 与 G1/G2/G4 细节。新定理、信息合同、定标、窗口、噪声、G3、重标盒和预算只有一个活动版本，不保留旧版本为并列候选。

**审定模型　gpt-6-astra　｜　审定时间　2026-11-28 19:50（北京时间，UTC+8）**

**裁决**　采用有条件的支撑恢复定理、匹配信息的两阶段对照和下述两套静态定标。设计修订覆盖原 Reviewer 的 B1–B3、M1–M4、m1–m2。现有证据包括解析推导、基准 1 的参考矩阵谱和基准 3 的完整平衡点与雅可比谱。轨迹上的持续激励、强度占比、跃迁频率、恢复效果、运行耗时均待验证。执行仍为 `PAUSED_BY_USER`，模型指针 `default none`。本件是 v3.0 草案经审定后的正式 v3.1 设计；尚未完成独立 Reviewer 对本版本的复核。

本次实际指定模型的科学分工为 `/root/scientist`（定理与总审）、`/root/swing_calibration`、`/root/wc_calibration`、`/root/comparison` 及 `/root/motion_control`。v3.0 原稿的作者、模型和日期由本文历史来源及历史失败回执保留。本次审定不构成独立 Reviewer 对 v3.1 的复核。其历史输入为 v3.0 的审查报告、暂定 Scientist 原稿、val 轨迹核查与架构简报，版本由原审查清单固定，部署合同见 `kiki.md`。

## 1　B1 · 从运动可知的量与 G3 所检验的量

用 $\mathcal O_0$ 专指一条事件的 11 帧、100 ms 因果历史与事前冻结常数。$\mathcal O_0^{\rm traj}$ 在训练时加上同一系统所有合法训练事件的完整运动（振子的 $\theta,\omega$ 或神经群体的 $r$）。$\mathcal O_1$ 再加训练期聚合端口；$\mathcal O_2$ 才含真图、真实隐态、分项、生成器激励和真系数。所有部署预测均从 $\mathcal O_0$ 自主推进，保持观测末帧及共同参考。训练缺帧和真实截止后不填充。预测、初始化与端口分解沿用同一方程。

**Theorem 1，结论 4a（端口规范）。** 已知 $M,D$、连续无噪运动时可求 $n=M\dot\omega+D\omega$。观测端口的线性算子 $[I,-I]$ 的秩为 $N$，核为 $\{(v,v)\}$，逐时刻恰 $N$ 维。允许端口函数自由变化时各时刻可有不同的 $v(t)$。在允许同一衰减模态的受限闭合子族，取 $\tau_m=p,\ \tau_e=F(\theta)+e$，$\dot p=-\lambda p,\ \dot e=-\lambda e$，$p_0,e_0$ 同加 $v$，则两端口同加 $e^{-\lambda t}v$，对整条轨迹给出 $N$ 个独立自由参数。物理先验若将其中一端完全指定，可能缩小相应参数族内的核。这里的必然性指运动观测算子的核，以及允许类上的非单射性；没有把任意 $\delta(t)$ 都算作某个固定异衰减 5N ODE 的解。WC 的端口和映射之核为 $(v,-v)$。

**结论 4b（静态骨架）。** 同噪声、同激励、同窗口下考察两个实验。振子对 $\phi=\Phi'\in C_c^2$ 有
$$
 y_{\rm net}=\langle\phi',M\omega\rangle-\langle\phi,D\omega\rangle
             =\langle\phi,\tau_e-\tau_m\rangle .
$$
对有已知重力项 $-G\sin\theta$ 的摆阵，左端还须减 $\langle\phi,G\sin\theta\rangle$。恒等式不需求观测的数值导数。若 $\tau_m$ 也在可消去的局部记忆类中，且下节秩成立，纯运动可有与端口 Stage 1 相同的静态解。对不在 $\mathcal F_{\rm env}$ 的快变机械驱动，须区分两种结果。若允许的驱动差与静态列空间在已观测轨线上有非零交集（包括将未知边力原样移入驱动），才有该方向的精确歧义；若低维驱动可另外剖除且余下矩阵满秩，仍可恢复，只可能改变精度。带宽越宽本身不是不可辨识证明。

在 $\|\phi\|_2=1$ 的同形频带窗中 $\|\phi'\|_2$ 的尺度为 $O(\Omega_k+L_k^{-1})$。速度噪声产生 $\langle\phi',M\delta\omega\rangle$，其最坏误差多出该因子；离散白噪声的方差按实际求积权重平方和算。端口通道直接取 $\langle\phi,\delta\tau_e\rangle$，两法还共同受到角度、速度设计阵误差影响。噪声说明估计的条件性与精度，不推出无噪观测不单射。基准 3 的反演还要求 $s=(\tau_r\dot r+r)/(1-\alpha_r r)\in(0,1)$，且 $(S^{-1})'(s)=1/[\gamma_s s(1-s)]$。反演只给出两电流之和；噪声在近饱和处放大、越域时无实值。非线性与弱积分一般不交换。两类场景均开放给完整的纯运动对照，数值差距由实验裁决。

**G3 规格。** 基准 1 `strong_open` 和基准 3 $N=8$ 各用相同三种子配对 `PDD-Frozen` 与 `No-Privilege Two-Stage`；`No-Privilege Same-Arch` 是次级对照，用于分辨端到端优化与信息因素。基准 1 全轨迹噪声两侧格亦加入 Two-Stage。两级目标、结构选择、256 步课程、训练事件、部署条件和评分保持可比；无特权方法取得全合法运动、相同一致性损失及已公开驱动字典。主比较沿用原 G3 的预登记效应量：三种子逐种子分别检验 AUPRC、F1 各提高至少 0.15，以及分离 RMSE 相对降幅至少 35%；预测 MSE 降至少 15% 或失稳 AUROC 增至少 0.05，另一项按原护栏控制（MSE 至多相对恶化 5%，AUROC 至多相对降低 5%）。分别报告三个终点，只有全过才使用联合主张；不以分离收益代替骨架收益。各项中的种子波动和事件内误差分别报告。该门禁比较同时去除两阶段端口信息后的总效果，不能据此分别量化 Stage 1 标签与 Stage 2 标签的因果贡献。无特权追平时收缩经验主张，结论 4a 的代数结果仍成立。配对单位是事件或来源事件簇；同一真图上的 F1/AUPRC 按图与优化种子逐项列值，不给边重采样伪造的跨系统置信度。两阶段主比较另外给同轨迹/同种子的配对误差。基准 3 单列跨方程族结果。

## 2　B2、M1 · Theorem 1 的设计矩阵、有限带宽误差与支撑恢复

**A1–A2。** 取一个固定参考、有限简单无向候选图、定向关联 $B$，$\varphi=B^\top\theta,\eta=B^\top\theta_\star$，$s=|B|^\top a$。在振子系统，互易项为 $B[(c^\star+\beta^\star\tanh s)\sin\varphi-c^\star\sin\eta]$，可选损耗项为 $|B|[g^\star(\cos\varphi-\cos\eta)]$。逐边乘法，$0\le\beta^\star\le0.8c^\star$。环境端口为 $S_e e+h(a)$，$h$ 节点局部，无瞬时边差模直通。写 $\dot e=-\Lambda_e e+\Gamma_e\omega+b_e,\ \dot a=-\Lambda_a a+\Gamma_a\omega+b_a$；两 $\Gamma$ 为常对角阵，$b_e,b_a$ 无显式速度项且在有限窗内平方可积。非线性或非对角速度反馈若另加，须重建干扰空间及其秩。此项声明限制对高阶约化的外推。

**B2(c) · 一个加权问题。** 用 $R$ 表事件数，$K$ 表每事件的测试函数数。每个合法完整窗的原始 $N$ 行为
$$
 X_c=B\,\mathrm{diag}\langle\phi,\sin\varphi-\sin\eta\rangle,\quad
 X_g=|B|\,\mathrm{diag}\langle\phi,\cos\varphi-\cos\eta\rangle,\quad
 L=\mathrm{diag}(-\langle\Phi,\omega\rangle),\quad y=\langle\phi,\tau_e\rangle .
$$
全局和采用 $w=\sqrt N/2,\ A_w=[I_N;\,w\mathbf1^\top]$。对每块的 $y,X_c,X_g,L$、其他正式剖除列以及一切误差**同时**左乘 $A_w$。令 $A$ 为重复的块对角算子。分母 $n=K_{\rm aug}=RK(N+1)$，未增广 $n_0=RKN$；有掩码时用实际完整块数替换 $RK$，不拼凑缺节点的全局行。用秩揭示 SVD 得 $Q=I-(AL)(AL)^+$，并令 $X=QA[X_c,X_g],\ Y=QAy$。剖除额外列时并入 $L$。固定浮点秩容差 $\max(\#\mathrm{rows},\#\mathrm{cols})\,\epsilon_{\rm mach}\sigma_{\max}$，在实现报告数值残余 $\|QAL\|$；带岭逆不满足精确消去恒等式。增广行重复使用同一测量，协方差随之变为 $A\Sigma A^\top$；单独复制一份噪声会改变问题。

**A3 与 Step 1。** 令 $\Psi_{\cos}$ 含逐窗余弦系数，$P_d=I-\mathbf1\mathbf1^\top/N$ 逐块作用。增广前划分共模与差模，再加全局行，即有
$$
 \|A(X_c u+X_g v)\|^2=(4/N+4w^2)\|\Psi_{\cos}v\|^2+\|X_cu+P_dX_gv\|^2 .
$$
故若 $\Psi_{\cos}/\sqrt{RK}$ 与 $X_c/\sqrt n$ 的最小奇异值分别至少 $g_c,g_s>0$，$m_g=\|P_dX_g/\sqrt n\|$，记 $h_c=(4/N+N)g_c^2/(N+1)$，则加权静态块的最小奇异值平方下界为 $\min\{h_c g_s^2/(h_c+2m_g^2),h_c/2\}$（对第二项用 Young 不等式）。仅含正弦时直接用相应块。再乘静态列空间与干扰列空间最小主夹角的 $\sin\theta_{\min}$，得到投影奇异值下界。**实际条件是 $\gamma=\sigma_{\min}(X/\sqrt n)>0$**；有条件稀疏恢复可改用下文的活跃块条件。无调制、去干扰后零误差时给唯一系数。有剩余记忆时给有限误差界。

完全图小角、等权且零共模时，$c_0(NI-\mathbf1\mathbf1^\top)-Nc_0I=-c_0\mathbf1\mathbf1^\top$ 在 $\mathbf1^\perp$ 上为零，静态与节点局部干扰列可以重合；有限大摆角下须对非线性阵重新检验。A3 要求共模速度有**独立于差模的充分激励**（保留原始完整速度），或由大角非线性提供可测的列空间分离。只检测到非零平均转速不能推出满秩。在 `static_closed`、`strong_mod_closed` 两格，逐折、全训练分别记录带/不带 $L$ 的数值秩、$\gamma$、$\sin\theta_{\min}$、归一化阵的 $\kappa_2$，并列共模/差模弱速度的 RMS；理论按实际拟合的阵判断。若已知模型合同显式关闭某块，仅在该合同下省略。零列、秩亏或噪声超过下述恢复余量均标未获资格，不能用真图选择投影或删列。非二分图的识别来自上述时间与空间的联合结构。

**B2(a) · Step 2。** $h_i=\alpha_{1i}\tanh a_i+\alpha_{2i}\tanh^2a_i$，取 $\ell_i=h_i'(0)$。积分分部使 $S_e\Gamma_e$ 和 $\mathrm{diag}(\ell)\Gamma_a$ 对应的速度项精确落入 $L$。留下的局部读出为
$$
 \langle\Phi,h_i'(a_i)(\lambda_{ai}a_i-b_{ai})\rangle
 -\gamma_{ai}\langle\Phi,(h_i'(a_i)-\ell_i)\omega_i\rangle .
$$
记 $\bar E_v^2=\sum_{r,k}\|v^{(r)}\|_{L^2(I_k)}^2/n_0$，$C_\Phi=\max_k\Omega_k\|\Phi_k\|_2$；对新窗按其原函数实算。未增广、去掉上述常系数速度项后，记忆残差 RMS 的有限带宽上界为
$$
 B_{\rm mem}={C_\Phi\over\Omega_{\min}}\big[
 \|S_e\|(\|\Lambda_e\|\bar E_e+\bar E_{b_e})
 +\|h'\|_\infty(\|\Lambda_a\|\bar E_a+\bar E_{b_a})
 +\|\Gamma_a\|\,\bar E_{(h'(a)-\ell)\odot\omega}\big].
$$
最后一项是时变导数乘速度的泄漏地板项（残差上界中的保留项，未断言对每条轨迹存在正下界）。当激励速度随 $\Omega$ 增长时它可保持 $O(1)$；固定轨线提高窗频，$\gamma$ 又可随信号谱衰减。相同未增广 RMS 下，共模确定性残差的最坏放大为 $C_N=\sqrt{N/(N+1)}\sqrt{1+N^2/4}$，独立同方差零均值节点噪声在投影前的**期望均方根规模**（均方取根）为 $D_N=\sqrt{N/(N+1)}\sqrt{1+N/4}$ 倍。前式由 $\|A_w\|^2=1+w^2N$，后式由 $\mathrm{tr}(A_w^\top A_w)=N(1+w^2)$。相关噪声按实际协方差或最坏界。余弦块分母改善伴随分子放大。

采用实测的 $\widehat X,\widehat Y,\widehat Q$，把观测目标误差、列扰动乘真系数、干扰列误差乘真干扰系数、调制残差及离散求积误差一并计入 $r=\widehat Y-\widehat X\vartheta_{\rm target}$。满秩 LS 有
$$
 \|\hat\vartheta-\vartheta_{\rm target}\|_2
 \le {\|\Pi_{\widehat X}r\|_2\over\sqrt n\,\sigma_{\min}(\widehat X/\sqrt n)} .
$$
干净无调制时，记忆贡献至多 $C_NB_{\rm mem}/\gamma$。**趋零推论**仅在 $h$ 线性（其常数速度项可投掉）或 $\partial_\omega\psi_a\equiv0$ 的子族给出：沿一列宽带激励，若 $\inf\gamma>0$、其余窗能量与 $C_\Phi$ 一致有界、调制为零且噪声和求积误差趋零，则记忆造成的参数误差趋零。固定采样率还受 Nyquist 限制；此推论不用于任一固定轨迹或固定 401 帧数据。

**M1 · 结论 3 与实际算法。** 设 $d_j=\|X_j\|/\sqrt n,\ D_c=\mathrm{diag}(d)$，去掉并记录零列，$Z=XD_c^{-1}$，$q=D_c\vartheta$。求解
$$
 \min_{q\ge0}\|Y-Zq\|^2/(2n)+\lambda\mathbf1^\top q .
$$
在物理坐标这恰是 $\lambda\sum_j d_j\vartheta_j$。$S$ 为真活跃的正弦及已启用余弦索引。令 $v$ 为真实动态调制项经相同弱窗、增广与投影后的向量，$G=Z^\top Z/n$。定义
$$
 q^{\rm eff}_S=q^\star_S+G_{SS}^{-1}Z_S^\top v/n,\qquad
 q^{\rm eff}_{S^c}=0,\qquad c_e^{\rm eff}=q^{\rm eff}_{c,e}/d_{c,e}.
$$
这是活跃空间上的联合投影；仅用于定义估计对象或事后真值评分。$g$ 同估时亦可吸收投影。剩余 $v_\perp=(I-P_S)v$ 正交于 $Z_S$。可报告 $\mu_e^{\rm eff}=(c_e^{\rm eff}-c_e^\star)/\beta_e^\star$（$\beta_e^\star=0$ 记未定义，并单列交叉投影偏移）。瞬时刚度的 $[0.2,1.8]c^\star$ 界不能传给多列弱式投影；单边无混叠且正权时才是凸加权均值。$a$ 边际对称也不能推出 $Z_S^\top v=0$。精确支撑的前提包括 $q^{\rm eff}_S>0$，不能由投影定义循环证明。

令 $Y=Zq^{\rm eff}+r,\ b_S=Z_S^\top r/n,\ P_S=Z_SZ_S^+$。若 $G_{SS}\succ0$，活跃解为 $q^+_S=q^{\rm eff}_S+G_{SS}^{-1}(b_S-\lambda\mathbf1)$，其余为零。精确的一侧 KKT 判据是 $q^+_S>0$ 且对每个 $j\notin S$，
$$
 Z_j^\top(I-P_S)r/n+\lambda G_{jS}G_{SS}^{-1}\mathbf1\ \le\lambda .
$$
非活跃严格不等号保证解唯一。可检验的充分条件为 $\max_{j\notin S}G_{jS}G_{SS}^{-1}\mathbf1\le1-\eta_0$、非活跃投影残差上界 $<\lambda\eta_0$、$\min q^{\rm eff}_S>\|G_{SS}^{-1}(b_S-\lambda\mathbf1)\|_\infty$。这给出不可表征性、噪声与最小信号三个独立余量。实际阈值还要求每个活跃项在本块内 $q^+_j/\max q^+\ge0.05$；空块输出空支撑。去偏 NNLS 在选中列上单独解并验证 KKT 与正性。惟有这些条件满足时才说恢复精确 $S^\star$ 与估计投影刚度。跨格的 F1 仍是待测结果。

## 3　M1 · Theorem 2 与 Theorem 3 的适用域

| Theorem 2 的情形 | 可复核的算子与结论 |
|---|---|
| 1(a)、1(b) 零角激励 | 共同参考平衡使 $(c,\beta)$ 灵敏度为零。纯幅值见证另需 $\eta=0,\ h=\tau_{\rm env,cm}\equiv0,\ \partial_a\psi_e\equiv0$，且 $p=e=\omega=0$ 时相应右端为零，$a$ 子系统在该曲线上不依赖待判系数。线性及非线性 $a\to e$ 通路都须关闭。于是任意合法 $a(0)$ 可纯幅值演化，秩为 0，参数内点局部维数 $2\lvert E\rvert$（边界为可行锥）。 |
| 2 单空间模态 | 取 $\theta=\theta_\star+u f(t)$，边差恒满足所有环路约束。小角线性伴随模型有 $\alpha_e=\cos\eta_e(u_i-u_j)$，$y=(B_S\mathrm{diag}(\alpha)c+q)f$；$\mathbf f\ne0$，$q\in\mathbb R^N$ 为允许的同频环境响应。$\mathbf f\otimes[B_S\mathrm{diag}(\alpha),I]$ 的秩恰 $N$，零空间维数 $m=\lvert S\rvert$，任意连通图均可构造。比如线性系统在节点输入可选时用相应净力驱动该模态，或取合法自由模态。有限幅度正弦的高次谐波可打破等价。 |
| 3 瞬时环境混叠 | 放开相同边特征直通后，$[X_c,X_c]$ 仅见两组系数之和。若 $X_c$ 秩为 $k$，核维数 $2\lvert E\rvert-k$；满列秩时为 $\lvert E\rvert$。 |
| 4 无端口观测 | 逐帧秩 $N$、核维数 $N$，以及在同衰减子族内的整轨 $N$ 维见证，见结论 4a。 |

**Theorem 3。** 在有限图、有限系数、$M\succ0$ 下，指定 5N 振子闭合由线性项和有界且导数有界的函数构成；$K(a)\sin\varphi$ 两因子及导数均有界，下述快驱动亦有界光滑。故该具体场全局 Lipschitz，存在 $A,L$ 使 $\|f(z)\|\le A+L\|z\|$。局部 Lipschitz 与此增长界已足以延拓所有有限时解。一般更广闭合不能仅由线性增长推出全局 Lipschitz。有界正刚度不保证大角势 Hessian 正定、长时有界、渐近稳定或任意积分步长稳定。

在**历史含储能梯度反馈的幅值方程族** $\dot a=-\nabla_a U_c-\Lambda_a a$，原二机不变子系统 $\dot z=(\pi/2)K'(z)-z,\ z(0)=0$ 对 $K=e^z$ 满足 $\dot z\ge(\pi/2-1)e^z$，有限爆破上界 $(\pi/2-1)^{-1}$ s。对 $K=1+0.8\tanh z$，$0\le z(t)\le0.4\pi(1-e^{-t})<1.256638$。两核比较来自同一个梯度闭合。v3 的 $\dot a$ 不含 $K'$；该例只证历史族中存在会爆破的成员。WC 的域与最终有界性证明见 §5。

## 4　B3 · 基准 1 的物理系统及频带

**系统固定法。** 每个 $N=8,16,32$ 固定一个原生物理系统，各工况、观测噪声、优化种子复用。下表是这三个确定性系统的结果。生成器采用 NumPy PCG64，独立子流 $G_k=\mathrm{default\_rng}(\mathrm{SeedSequence}([20261128,N,k]))$。构造环边及弦 $(0,2)$，从按字典序排列的其余非环非自环边用 子流 $G_0$ 的 `choice` 无放回取 $N/2-1$ 个；将全部 $i<j$ 的边排序。$B_{ie}=+1,\ B_{je}=-1,\ C=|B|$。$G_1,G_2,G_3,G_4$ 依序分别取均匀 $m_i\in[.7,1.4],d_i\in[.12,.25],c_e\in[1.2,2.4],\theta_{\star i}\in[-.35,.35]$，参考角统一减节点均值。设
$$
 K=B\,\mathrm{diag}(c\cos\eta)B^\top,\quad
 s_N={\lambda_2(\mathrm{diag}(m)^{-1/2}K\,\mathrm{diag}(m)^{-1/2})\over(2\pi\cdot.8)^2},\quad
 M_i=s_Nm_i,\quad D_i=\sqrt{s_N}\,d_i .
$$
用子流 $G_5$ 的 `choice` 从排序后边序号无放回取 $N$ 条；按返回顺序赋 $G_6$ 生成的 $\beta_e/c_e\in[.65,.8]$，其余置零。$G_7$ 对所有真边给 $g_e/c_e\in[.25,.50]$。这些是生成数值；任何学习方法均不得从种子或生成器重建真值。$M,D$ 作为既有可测常数给各法。

| $N$ | 真边 / 候选 | $s_N$ | 全部 $N-1$ 个正自然频率的最小–最大值 (Hz) | 含 $D$ 的静态骨架振荡频带 (Hz) |
|---:|---:|---:|---:|---:|
| 8 | 12 / 28 | .083169103975 | .800000–1.832886 | .799146–1.831617 |
| 16 | 24 / 120 | .027718793350 | .800000–3.307230 | .796749–3.304417 |
| 32 | 48 / 496 | .015374441480 | .800000–4.503304 | .794143–4.500774 |

计算为 $\sqrt{\lambda(\,M^{-1/2}KM^{-1/2}\,)}/(2\pi)$，仅排除共同角度的零根。谱配准是已知真值生成器的定标步骤。两种封闭格令 $\tau_m=\tau_{\rm env}=g=0$，静态格 $\beta=0$，调制格保留 $a,\beta$；非活动隐态不进入运动方程。闭格在生成初始历史之前，除独立差模扰动外，对各事件全节点叠加同一事前抽取的共模初速（均匀 $[-.4,.4]$ rad/s、与差模和结局独立），此后按原方程演化；不对生成好的轨迹作后处理。合法历史能观测这项激励。两种开放格均开启调制、机械及环境，另在 `strong_open_sym` 开启 $g$。以 $\Delta_e=\cos(\varphi_e-\eta_e)-1$ 取代隐态驱动中的 $\cos\varphi_e-\cos\eta_e$，保留后者用于损耗端口。这使大摆角负反馈保留而参考处该驱动一阶导数为零。最终真值为
$$
\begin{aligned}
 2\dot p&=-.525p-.35\omega-.15\tanh p,\\
 2\dot a&=-.45a+.575\omega+.4C\Delta,\\
 2\dot e&=-.265e+.475\omega+.35\tanh a+.225C\Delta,\\
 \tau_e&=B[(c+\beta\tanh(C^\top a))\sin\varphi-c\sin\eta]
       +C[g(\cos\varphi-\cos\eta)]+1.8e+.35\tanh a,\\
 T_i(\omega)&=\tanh^3[(\omega_i-\bar\omega_M)/(.4\,{\rm rad/s})],\quad
 \bar\omega_M={\mathbf1^\top M\omega\over\mathbf1^\top M\mathbf1},\qquad
 \tau_{m,i}=p_i-b_iT_i(\omega),\quad b_i=.4 .
\end{aligned}
$$
$M\dot\omega=\tau_m-\tau_e-D\omega$。快反馈在参考的一阶导数为零，对当前差速有瞬时响应，依赖全网共模速度；它不属于 A2 的严格真节点局部慢端口类。对非零正弦差速，三次非线性产生高次谐波。例如幅值恰为 .4 rad/s 的单频探针，$T_i$ 基频、三次、五次幅值分别为 .37757、.08191、.02069。幅度 $.4$ 是生成真值；正式基线和 PDD 都享有相同模板、可学习 $b_i$，参考度量不暴露真图。纯运动 Stage 1 在 $L$ 加入 $\mathrm{diag}\langle\phi,T\rangle$ 剖除未知 $b$（因为净目标中符号为 $+bT$），再按 §2 完成同一套 FISTA/NNLS 与 Stage 2。学得的 $b$ 可作初始化，所有同构法在各自 Stage 2 都可训练 $b$。训练真值除测量外不充当未来驱动。控制组也能解释该快通道，实际可见的宽带强度仍须测量；应同时报告快/慢机械 RMS 和观测误差。修改后的 $\Delta$ 属于原正余弦静态字典的线性组合，或可直接作为公开候选输入，勿把参考重写进 $g$ 列。

**完整线性化交叉核验。** $R=M^{-1}$，$K_s=K-C\,\mathrm{diag}(g\sin\eta)B^\top$，$H=B\,\mathrm{diag}(\beta\sin\eta)C^\top+.35I$。按 $(\theta,\omega,p,a,e)$ 的解析 Jacobian 为
$$
 \begin{pmatrix}0&I&0&0&0\\-RK_s&-RD&R&-RH&-1.8R\\
 0&-.175I&-.3375I&0&0\\0&.2875I&0&-.225I&0\\
 0&.2375I&0&.175I&-.1325I\end{pmatrix}.
$$
表中 $\alpha$ 为仅除去唯一共同角度零根后的谱横坐标（$\mathrm{s}^{-1}$）。开放格各自出现两个分离的簇，以 $|\Im\lambda|>1\ {\rm s}^{-1}$ 标快振荡，分离倍数为最小快 $|\Im\lambda|$ 除以其余根最大 $|\lambda|$。包括共模转速；关损耗 / 开损耗按同一顺序列出。

| $N$ | 两封闭格 $\alpha$（静态 / 调制） | 两开放格 $\alpha$ | 两开放格全部快振荡频带 (Hz) | 两开放格快慢分离 |
|---:|---|---|---|---|
| 8 | −.240297 / −.204042 | −.088099 / −.083010 | .445854–1.881642 / .474400–1.904585 | 8.447 / 8.986 |
| 16 | −.450913 / −.211405 | −.057615 / −.055118 | .769320–3.397460 / .780550–3.446595 | 14.559 / 14.771 |
| 32 | −.559792 / −.207747 | −.038956 / −.038446 | 1.034524–4.604735 / 1.078445–4.669514 | 19.554 / 20.385 |

封闭格复算在活动态上做：使用上式的运动块及适用的 $a$ 行，断开机械、电气环境读出。频率单位一律 Hz；谱实部和慢极点单位 $\mathrm{s}^{-1}$。严格环境记忆 $e$ 的截止为 $.1325/(2\pi)=.02109$ Hz；即取 $p,a,e$ 三个孤立参考衰减中最快的 $.3375/(2\pi)=.053715$ Hz，最低正自然频率仍高 **14.893 倍**，最低完整开放振荡频率也高 **8.300 倍**。此处 “截止” 描述线性通道，不能代替 B2 的非线性速度泄漏界。

**Stage 1 主通带。** 在 0–4 s 取三个 2 s 窗，起点 $0,1,2$ s。每窗用 $f=.50,.75,\ldots,5.00$ Hz、$q=0,\pi/2$ 双相；总数 $K=114$。$u$ 为窗内时间，$\Psi=\sin^4(\pi u/2)\cos(2\pi fu+q)$，$\Phi=\Psi/\|\Psi'\|_2,\ \phi=\Phi'$，窗外零。至三阶端点消失保证 $\phi$ 的零延拓属于 $C_c^2(\mathbb R)$。强度以实际窗频响定义，紧支窗并非理想带通。求积用 $\Delta t=.01$ s 的梯形权；可从离散 $\phi$ 扣除 $b(u)\sum w_t\phi/\sum w_tb$、再归一（$b=\sin^4(\pi u/2)$），同时对 $\phi'$ 做同一解析修正和缩放；相应离散误差纳入 §2。新窗不套用旧 $\sin^2$ 包络的常数。

复算每行 $k$ 的 $H_k(f)=|\sum_t w_t\phi_k(t)e^{2\pi ift}|$，在 $0:.0005:5.5$ Hz 上求行峰 $P_k$，取 $\max_k H_k(f)/P_k$。在 **[.44,5.00] Hz** 主带网格下界为 **.74019657**。以窗中心为相位原点，行归一化频响的 Lipschitz 上界 $\max_k 2\pi\sum_t|t-1||w_t\phi_k(t)|/P_k=2.41093\,{\rm Hz}^{-1}$；插值间隙最多 .00025 Hz，故整段下界 $.73959>1/\sqrt2$。该主带覆盖三个尺寸的全部正自然频率及表中开放振荡。离散 DC 残差 $<10^{-15}$。若事件被截断，仅使用观测完整的窗、登记丢窗率和缺事件，尤其检查快失稳事件对 A3 的贡献；不能把过早截止的事件当作有 114 个窗口。

**Run 0 分支 B 的替换与重校准。** 删除 “删 0.5 Hz” 及其变体。支撑差时依次报告完整窗覆盖、实际谱能量、A3 夹角/条件数、噪声与冻结后可查的 $\mu_e^{\rm eff}$、残差及核验分母。强失稳下不设 $\mu_e=0$。`PDD-Frozen` 使用同一 $\hat S,\hat c,\hat g$；`Stage-2 Bounded-Recal` 保持 $\hat S_c,\hat S_g,\hat g$ 不变，在每次更新后对 $e\in\hat S_c$ 投影
$$
 .50\,\hat c_e\le c_e\le2.20\,\hat c_e,\quad 0\le\beta_e\le.8c_e ;
 \qquad c_{e\notin\hat S_c}=\beta_{e\notin\hat S_c}=0 .
$$
若 NNLS 把某选中边估到零，记录支撑/非零权不一致；该盒在该边退化为零，不偷偷补边。盒覆盖某真系数的充要条件是 $\hat c/c^\star\in[1/2.20,2]$。例如就算在理想单列下 $|\mu|\le.7$，$\beta/c=.8$ 仍允许 $c/\hat c=1/.44=2.2727$。故此盒是有预算的消融约束，不能声称必定含真值。冻结/重标并列用同一骨架、数据和种子，报告 $c^{\rm eff}$ 相对误差、相对 $c^\star$ 误差、越盒率、触边率、端口、MSE、VPT 及失稳。评分用的真值不回流折内拟合；一次冒烟差异不能晋升模型。

## 5　B3 · 基准 3 的稀疏 Wilson–Cowan 定标

**拓扑与真值。** $N=8,16$，$n_E=3N/4,\ n_I=N/4$；编号从 0 开始，$W_{ij}$ 表示 $j\to i$ 的有符号权。每个接收 E 节点 $i$ 从 $(i-1)\bmod n_E,(i-2)\bmod n_E$ 各取 $+1.25$，从 I 节点 $n_E+\lfloor i/3\rfloor$ 取 $-1.5$。每个接收 I 节点 $n_E+k$ 从 $3k,3k+1,3k+2$ 各取 $+2/3$，从 $n_E+(k-1)\bmod n_I$ 取 $-0.5$。置换节点时连同类型、常数、数据一起置换。真边数为 26/56、52/240。学习器只得有向无自环完全候选图与已知源类型，不能抄生成器的连接规则。以下四块按**接收行、发出列**消除下标歧义。

| 接收 ← 发出 | 有符号行和 | $\beta_{ij}/w_{ij}$ 真值，$w=\lvert W\rvert$ | Stage 2 可学区间 |
|---|---:|---:|---|
| E ← E | +2.50 | +0.80 | $[0,0.8]w$ |
| E ← I | −1.50 | −0.02 | $[-0.8,0]w$ |
| I ← E | +2.00 | +0.05 | $[0,0.8]w$ |
| I ← I | −0.50 | −0.02 | $[-0.8,0]w$ |

令 $d=r-r_\star,\ r_{\star i}=0.15,\ \tau_r=0.08\,{\rm s},\ \alpha_r=0.15$，$S(x)=(1+\exp[-5(x-0.5)])^{-1}$，$I_0=S^{-1}(.15/(1-.15^2))=0.158445237369$。令 $C_{ij}=\sigma_j\beta_{ij},\ K_{ij}=W_{ij}+C_{ij}\tanh(a_i+a_j)$。精确生成方程为
$$
\begin{aligned}
 I_{\rm drive}&=u,\quad I_{\rm net}=Kd-1.5m+.01\tanh a,\\
 .08\dot r&=-r+(1-.15r)\odot S(I_0+u+I_{\rm net}),\\
 \dot u&=-.6u-.02d,\qquad
 \dot a=-.5a+2d-.1\tanh a,\qquad
 \dot m=-.3m+.02d+.01\tanh a .
\end{aligned}
$$
六项 $(\gamma_{ur},\gamma_{ar},\gamma_{a,\mathrm{sat}},\gamma_{mr},\gamma_{ma},\alpha_{\mathrm{cm}})=(.02,2,.1,.02,.01,.01)$，所有节点取同值。高发放提高 $a$，E 源权幅值增加，I 源抑制幅值轻减；所有突触全程保持源类型符号。该 STP 为有效突触可塑性模型。确定双稳态的代价是将原稿适应与调制的 RMS 占比、强调制边比例撤为待测量，不据静态根宣称满足这些比例。

**静态证明与复算。** 群体均匀子空间因四块入权行和固定而不变。逐类令 $u=-d/30$、解 $.5a+.1\tanh a=2d$（严格单调），代入 $m=(.02d+.01\tanh a)/.3$、$\dot r=0$ 得二维不动点方程。全图稳定性用完整 $4N$ Jacobian，未截去空间模态。以下按 E/I 顺序给出平衡坐标。

| 平衡 | $(r_E,r_I)$ | $(u_E,u_I)$ | $(a_E,a_I)$ | $(m_E,m_I)$ |
|---|---|---|---|---|
| 低发放 | (.150000000, .150000000) | (0, 0) | (0, 0) | (0, 0) |
| 中间 | (.194471533, .190435918) | (−.001482384, −.001347864) | (.148418490, .134921856) | (.007876043, .007166030) |
| 高发放 | (.869436721, .846807657) | (−.023981224, −.023226922) | (2.679619866, 2.589471587) | (.080983618, .079413684) |

| $N$ / Jacobian 维数 | 低态最大实部 | 高态最大实部 | 中间态不稳定根数 |
|---|---:|---:|---:|
| 8 / 32 | −0.316364420 | −0.300021468 | 1 |
| 16 / 64 | −0.297922487 | −0.300021468 | 3 |

两尺寸在低态的 E/I 共轭对都是 $-4.833973665\pm6.893986588\,i\ {\rm s}^{-1}$，阻尼振荡频率 **1.097212043 Hz**。它是一对群体模态，并非要求全谱的主导根为复根。中间态均有正实根 $+5.224949837$，$N=16$ 另有 $+4.211062378\pm26.110917358\,i$。高态 $\bar r_E>.65$ 且全谱严格左稳，给出两个局部吸引平衡的证据。不能从中间态推断全空间分界或从低历史必达高态。

复算处方：按上述稀疏规则建 $W,C$，取猜测 $r_i=t,\ u_i=-(t-.15)/30,\ a_i=4(t-.15),\ m_i=.1(t-.15)$，$t=.15,.20,.85$。在 float64 对 $F(z)=0$ 求根，解容差 $10^{-11}$，$F$ 取 $\dot r,\dot u,\dot a,\dot m$ 的拼接。一次按解析导数建矩阵；又独立用复步 $J_{\cdot j}=\mathrm{Im}\,F(z+10^{-20}i\,e_j)/10^{-20}$ 重算并求全部特征值。两尺寸三点此次复算的最大 $\|F(z)\|_\infty=2.64\times10^{-14}$ 以下。推导判稳依据为高低平衡的双曲性。这些是静态浮点根及余量，不是区间算术认证。

**定理 3 的 WC 域。** 在 $r=0$ 向内，在 $r=b=(1+\alpha_r)^{-1}$ 有 $\tau_r\dot r=b(S-1)<0$。$\mathcal D=[0,b]^N\times\mathbb R^{3N}$ 对其中初态严格向内、前向不变。较大的 $[0,1]^N$ 也不变；从 $r_i(0)>b$ 出发时只可写 $r_i(t)\le b+(r_i(0)-b)_+e^{-(1+\alpha_r)t/\tau_r}$。由于 $r$ 有界，$K$、$K'$、$S'$ 有界，完整场在该条带上全局 Lipschitz。$u,a,m$ 各为严格衰减加有界驱动，变常数公式给出与初态无关的最终上界。部署含噪时所有方法精确保留实测末帧，记录逐节点与逐事件越界率，不裁剪、不删除事件或替换干净末帧。盒外初态不适用盒内不变性；因 $S,\tanh$ 有界，整个 WC 场对任意有限初态局部 Lipschitz 且至多线性增长，仍有全局解。

**WC Stage-1-Only。** 与 PDD 共用仅从训练 $I_{\rm net}$ 求得的有向 $\hat S,\hat w$（Dale 符号已知），$K=\sigma_j\hat w_{ij}$ 固定，置 $\beta=a=m=0$，删除各自读出；保留完整正向 sigmoid、已知不应期和参考。仅训练 $\dot u=-\Lambda_u u-G_{ur}(r-r_\star)$ 的对角线性系数及 $u_0=I_\phi(\mathcal O_0)$ 的编码器，和 $r_{\rm last}$ 一起推进，沿用相同可得聚合电流监督、课程与预算。比较对象可预测非恒定外源电流。

**WC 纯运动 Two-Stage 的实现合同。** 与振子的净转矩分别按物理方程构造。WC 的正式 Stage 1 直接拟合正向 sigmoid 的弱式。对节点 $i$、窗口 $k$，拟合下列有限窗残差
$$
 E_{ik}=-\tau_i\langle\phi'_k,r_i\rangle+\langle\phi_k,r_i\rangle
 -\langle\phi_k,(1-\alpha_r r_i)S(I_{0i}+\sum_{j\ne i}\sigma_jw_{ij}d_j(t)+n_i(t))\rangle .
$$
这里 $d_j(t)=r_j(t)-r_{\star j}$；$n_i$ 是可观测运动支持的低维电流干扰。取 $\lambda_\ell=(.25,.60,1.20)\,{\rm s}^{-1}$，特征为 $e^{-\lambda_\ell t}(r_i(0)-r_{\star i})$、$e^{-\lambda_\ell t}[r_i(0)-r_i(-.1)]$ 和 $\int_0^t e^{-\lambda_\ell(t-s)}d_i(s)\,ds$。每节点九个常系数跨事件共享；没有逐帧自由电流。用相同训练历史、窗和候选图，目标为归一化弱残差平方均值的一半加 $\lambda\sum d_{ij}^{\rm rms}w_{ij}$ 与 $10^{-6}\|b_n\|^2/2$；$w\ge0$，标准化使用训练子折的列 RMS，取穿过 sigmoid 的弱响应在参考电流 $I_0$、观测回归因子 $r_j-r_{\star j}$ 下对 $w$ 的 Jacobian 列 RMS（包含 $S'(I_0)$）计算尺度。相对惩罚用 §7 的七点网格，基准梯度为 $w=0$、拟合干扰后的最大正向下降梯度；数值为零时保留空模型并报告。每折两种确定起点（$w=0$ 和每行均分总正幅值 1，干扰系数 0），有界迭代 L-BFGS-B 每起点至多 300 次；按训练目标择优，再以验证事件的前向弱残差选惩罚。0.05 阈值及固定支撑无惩罚 refit 同 PDD。Stage 2 换为完整同构 WC、严格冻结所得 $\hat w,\hat S$，仅用运动与一致性损失等训练 256 步。这项额外静态非凸优化与卷积计时代价单列，单种子四折训练至多 $2(4\times7+1+1)\times300=18{,}000$ 次优化迭代，另记录零字典干扰初拟合的实际迭代（每训练子集最多 300）。PDD 的 NNLS 收敛性质不移用到本法。

另报告纯运动逆读出的诊断：用固定 9 帧三次局部多项式（端点单侧、不跨缺测）估 $\dot r$，统计原始带噪运动中 $s\notin(0,1)$ 及 $s\notin[.02,.98]$ 的比例。正式前向弱式使用所有有完整窗口的事件、从不调用 logit；不得因逆读出越域把该方法记 NaN 或淘汰。不论谁胜，G3 只说明所定义方法及预算下的净收益。

## 6　M2 · 观测噪声、分层与尺度

| 合同项 | 冻结规则 |
|---|---|
| 噪声主轴 | 分别存 $\sigma_H$（部署与训练历史）、$\sigma_X$（训练全运动）、$\sigma_P$（训练聚合端口）。主横轴取三者同为 $0,.01,.03$ 的三格，主锚为 $(.01,.01,.01)$。每条合法 0–4 s 训练测量均按本通道加噪，含 $\tau_m,\tau_e$ 或 $I_{\rm drive},I_{\rm net}$。不沿用只扰动 11 帧的生成器。 |
| 可复现噪声 | 原始传感单位按 $\theta:1$ rad、$\omega:1$ rad/s、各转矩/电流/发放率各 1 个本合成系统单位固定；用 $\sigma$ 乘单位尺度的零均值独立高斯。每事件、传感器、节点、物理时间戳锁定同一随机抽样，各法共用，三档可共用单位正态底样。历史与全轨迹重合帧只加一次。生成器角度/速度真关系不强制回填至含噪观测。WC 的越域原始测量、末帧越界率和逆读出失效率留档；干净分项/标签仅评分器可读。 |
| 可归因比较 | 同一已拟合模型可固定全轨迹/端口训练噪声，配对重评不同历史噪声，作为部署敏感性副表，无需新拟合。主横轴三者联动，不能将差值完全归因于某一个噪声源。全轨迹弱式验证亦在相同有噪观测上计算，最终轨迹误差另对冻结真实未来评分。 |
| 划分与批次 | 按事件整组分折，来源相同的成对扰动或重复测量保持同组（分组键仅隔离，不作特征）。用训练运动可得的结局与历史能量分位分层，所有无特权对照也可计算。振子能量代理取历史平均 $\sum_i M_i\omega_i^2/\sum_iM_i$，WC 取平均 $(r_i-r_{\star i})^2$。折划分可先按可观测结局固定；批次的能量四分位及任何用于拟合的分位点只在训练子折估计，稀小层按相邻能量层合并。禁止读取 $\mathcal O_2$ 的 `excitation`、隐负荷规模或生成来源做模型/CV/批次选择。 |
| 环境尺度与初态 | 公开 `S_e_scale=1.0` 只定义模型有效转矩单位。基准 1 生成器 `env_gain_true=1.8` 仅构建真值；若另建异质生成格，其真值同样隔离。以 $e_{\rm model}=1.8e_{\rm gen}$ 可说明有效坐标，模型参数吸收尺度。学习态 $p,a,e$ 的编码器用参考扣减、有限线性输出，取消原 `e_init_bound=2` 等隐态硬截断；无特权版本不读取端口分位数。检查输出、尺度与饱和/尾部误差，不能从聚合端口推出真隐态范围。 |
| Stage 2 与部署 | 损失的运动/合法端口等尺度只用该系统该档带噪 `train` 计算并冻结，例如去共同参考后有效元素 RMS、下限 $10^{-6}$，每个物理通道独立。无特权方法只取运动尺度；部署所需归一化常数作为共享冻结模型参数。隐藏 metadata、端口、真图、分层标签不得进入初始化、模型 rollout、数据加载旁路或缓存。用替换特权/真值数据的污染副本核查输出逐比特不变。 |

## 7　M2、M3 · 拟合与基线规格卡

**四折无泄漏弱式。** 每个训练子折 $T$ 单独计算全部投影、$d_{j,T}$、噪声/损失尺度、相对网格基值与干扰系数。以 $X_T^A,L_T^A,y_T^A$ 表示已增广、未投影的行。先用 $Q_T$、归一化列拟合 $\hat\vartheta_{T,\alpha}$，再得 $\hat b_{T,\alpha}=(L_T^A)^+(y_T^A-X_T^A\hat\vartheta_{T,\alpha})$。验证用 $X_V^A\hat\vartheta_{T,\alpha}+L_V^A\hat b_{T,\alpha}$ 直接预测 $y_V^A$；验证目标不参与重估干扰、标准化、选列或投影。相对网格 $\alpha\in\{10^{-4},3\!\times\!10^{-4},10^{-3},3\!\times\!10^{-3},10^{-2},3\!\times\!10^{-2},10^{-1}\}$；零解临界值用归一化列的最大正梯度相关 $\max(0,\max_j Z_{T,j}^\top Y_T/n_T)$。四折每折对事件等权，选在最低均值加其一标准误内最大的 $\alpha$。最后用完整 train 重算所有量再拟合。WC 特权线性问题以逐接收节点带 Dale 符号的 $d_j(t)$ 建列，不复制振子全局和行、没有速度 nuisance。WC 无特权按 §5 用训练拟合的 $w,b_n$ 直接预测验证弱矩。所有实现将确切窗、截断、列与阈值写入后续配置。

| 方法 | 训练和图推断 | 部署、结构读出与公平边界 |
|---|---|---|
| `No-Privilege Same-Arch` | 端到端同构，完全读 $\mathcal O_0^{\rm traj}$，运动损失、$0.05\mathcal L_{\rm cons}$ 及登记稀疏项，256 步；无端口和起点端口监督。 | 与 PDD 用同一短历史编码器与闭合自由度。它与 PDD 的差含训练程序；不能单独裁决 G3。 |
| `No-Privilege Two-Stage` | 振子 §1、§4 的合法净力弱式，WC §5 的正向饱和弱式。与特权方法各自从完整训练运动做 Stage 1，共享支撑阈值、train-only 选参与同构 Stage 2。 | 冻结自估骨架、支持与损耗权重，Stage 2 同样享有 $\mathcal L_{\rm motion}+.05\mathcal L_{\rm cons}$ 及同构正则，无端口损失。$\mathcal L_{\rm cons}$ 只在训练时用下一段 0.1 s 观测重编码对齐隐态，部署不重编码。 |
| `NRI/ACD` 与 `NRI+Port` | 明示为固定系统的 NRI/ACD 风格适配。完整 train 轨迹经时序/边编码器产生每事件边 logit，按有效帧掩码、事件等权平均得系统共享 $q_{ij}=\mathrm{sigmoid}(\mathrm{mean}_r\,l_{rij})$。二值 Concrete/Gumbel 门、消息 ODE、train 重构与 Bernoulli 稀疏先验共同训练；CV 只池化训练子折。无向振子绑定双向门、WC 保留方向。固定默认先验边率 .25、KL 系数 $10^{-3}$、温度在课程内由 1 线性降至 .5。 | 部署冻结系统图，以 $q$ 的确定性软门做预测、连续图指标用 $q$，二值阈值 .5。事件历史只初始化隐态。`+Port` 在相同消息/节点解码器上监督参与 RHS 的两个聚合端口，基线只训运动。一般聚合读出不能当作内部/环境分离真值。 |
| `SINDy-AE` | 保持物理观测坐标、共享历史编码器只学隐坐标。候选字典用节点的线性、至二次、tanh 与预声明物理边组（振子 $\sin,\cos$ 中心化特征，WC 为有向 $r_j-r_{\star j}$）。运动与聚合端口共同训练。先密训 128 步，Phase B 完结才准首次剪枝；在 128、160、192、224 步作顺序阈值最小二乘/岭重拟合，最后 32 步固定支撑。按 train 子折定列 RMS；组分数取归一化贡献的二范数，首次及后续阈值为同类组最大分数的 .05。剪枝时在训练轨迹的合法历史窗逐时编码得到隐坐标，和实测运动组成停止梯度的 $z$；以 $-\langle\phi',z\rangle$ 为目标对 $\langle\phi,\Theta(z)\rangle$ 做阈值岭回归（列 RMS 归一，岭 $10^{-6}$），端口字典另以实测聚合读出拟合。支撑掩码之后继续端到端运动/端口训练；记录字典秩和求解成本。 | 只有明确标定端点 $(i,j)$ 的物理读出组贡献能映射成静态边分数。隐坐标之间的稀疏系数、节点自项均不能事后命名为线路。若采用无物理读出的 AE 实现，其拓扑指标填 N/A。 |
| `PHNN/LGNN` | 仅振子和摆阵适用，采用下述单一开放 PHNN 适配；WC 填 N/A。等容量区间约 2–3k 参数是设计目标，报告实际模型大小、潜态与计算时间。 | 完整 $5N$ 演化、初态、参与动量方程的端口和训练监督共用同一套参数；图势未声明逐边支持时，硬图指标 N/A。 |

**5N 开放端口哈密顿基线。** 用 $\xi=(q,\pi,h),\ q=\theta,\ \pi=M\omega,\ h=(p,a,e)$，取
$$
 H={1\over2}\pi^\top M^{-1}\pi+V_\psi(q,a)+U_G(q)+\tfrac12 h^\top C_hh,\qquad C_h=I .
$$
$V_\psi$ 为平滑图势，在共同参考扣除常数与一阶参考项；$U_G$ 为摆阵已知重力势或零。令 $v=\nabla_\pi H,\ \chi=\nabla_h H,\ R_h=L_\psi L_\psi^\top+10^{-6}I$（节点局部 $3\times3$ 块）。正式方程为
$$
 \dot q=v,\quad \dot\pi=-\nabla_q (V_\psi+U_G)-Dv
             +\hat\tau_m-\hat\tau_{\rm env},\qquad
 \dot h=-R_h\chi+j_\psi(\xi).
$$
等价于 $\dot\xi=(J-R)\nabla H+u_{\rm open}$，其中 $J_{q\pi}=I,J_{\pi q}=-I$、其余块为零，$R=\mathrm{diag}(0,D,R_h)$，$u_{\rm open}=(0,\hat\tau_m-\hat\tau_{\rm env},j_\psi)$。$j_\psi$ 为参考扣减的节点共享主动源，包含三组隐态动力学。机械读出可用 $S_p p-bT(\omega)$，环境读出为参考扣减的可学习状态头；$\hat\tau_e=\nabla_q V_\psi+\hat\tau_{\rm env}$（重力另计）。编码器从历史给 $p_0,a_0,e_0$，合法末帧给 $q_0,\pi_0$。采用 PDD 同口径的运动、聚合端口、起点端口与一致性监督。由链式法则，
$$
 \dot H=v^\top(\hat\tau_m-\hat\tau_{\rm env})-v^\top Dv-\chi^\top R_h\chi+\chi^\top j_\psi .
$$
最后一项记入开放功率账；该恒等式不授予整个系统被动性或 SI 能量解释。

## 8　M4 · 更新、复用与资源口径

原表各行是 $7+33+40+20+12+18+28=\mathbf{158}$ 个 ODE 训练条目，$158\times256+128=40,576$，原文写的 128 个有误。按本规格复用原本的 5 个基准 1、2 个 WC 的 Run 0 冒烟时，相同配置仅有 151 个独立拟合。v3.1 新增无特权双阶段独立拟合 10 个（基准 1 锚三种子、两端噪声各两种子、WC 锚三种子）；两个 Run 0 冒烟再次引用其中各一项。

| 阶段 | 新的独立 ODE 训练数 | 主课程更新 |
|---|---:|---:|
| Run 0，基准 1 六端点、WC 三端点 | 9 | 2,304 |
| Phase 1 基准 1 锚，$12\times3-6$ | 30 | 7,680 + 128 |
| 工况纵轴 | 40 | 10,240 |
| 全轨迹噪声两端 | 24 | 6,144 |
| 规模横轴 | 12 | 3,072 |
| 基准 2 | 18 | 4,608 |
| 基准 3 其余，$9\times3+4-3$ | 28 | 7,168 |
| **合计** | **161** | **41,344** |

修订毛计 **170** 项、去重 **9** 项；另有 12 个非 ODE 参照。振子 Stage 1 的 FISTA 仍以 2,000 步及投影 KKT 残差 $10^{-10}$ 为上限/收敛准则；Stage 2 的 256 步为 $64@1\,{\rm s}+64@2\,{\rm s}+128@4\,{\rm s}$。原稿训练 RK4 0.01 s、训练前 0.01/0.005 s 双网格、评分 0.005/0.0025 s 双网格作为未来资格保留。128 步追加为从无特权端到端锚格种子 24121 的第 256 步继续至第 384 步。该检查不能代替对无特权双阶段的优化健康报告。主课程计数不含 CV、非凸弱式求解、双网格梯度资格、失败重跑、打分或 dry-run。原 7 个及修订后 9 个冒烟可以复用拟合检查点的前提是代码、数据/噪声、超参、种子、课程与选点合同哈希相同；任何一项变更，另记实际成本及谱系。Run 0 的止损和实现适配用合成 train/val；冻结合成确认集留到合约锁定后评分。相关轨迹与优化种子不重复计作独立物理系统。

**Run 0 预算与止损。** 基准 1 六项为 `Stage-1-Only`、`PDD-Frozen`、`Bounded-Recal`、`End-to-End v2`、`GraphNeuralODE`、`No-Privilege Two-Stage`；WC 三项为 `Stage-1-Only`、`PDD-Frozen`、`No-Privilege Two-Stage`。两种 PDD 用同一份 Stage 1 结果。原 G1/G2 冒烟门槛继续用于合成验证，失利时保留结果并判断是观测条件、闭合还是优化预算所致。基准 3 还须从符合低历史条件的事件实测连续超阈值时长 $\ge.20$ s（用时间戳末减首，$.01$ s 网格需至少 21 个连续超阈值采样点，或明确计时的 20 个完整区间）。不把高平衡存在当作跃迁样本已足额。若 RMS 或覆盖不足，标对应原假说未获资格。任何调整形成新冻结版本，不能根据确认集改造生成器。

单进程 CPU float64、batch 4。未来 dry-run 对 `PDD`、`GraphNeuralODE` 各测 $N=8,16,32$；`PHNN` 在 $N=8$ 振子和摆阵各测一次，并在 $N=32$ 加一次内存压力探针；`No-Privilege Two-Stage` 在基准 1、WC 的 $N=8$ 各测一次。共 11 配置，每配置 1 个预热、3 个 4 s 计时批次，含完整梯度计算而不做 `optimizer.step`。另对双阶段记录实际 Stage 1 的建阵、4 折、峰值内存及 refit，WC 的两初值/解算成本独列。PHNN 必须测对 $\nabla H$ 再反传的二阶计算，记录前向、反向、峰值 RSS、是否越过原 4 GiB 资格线；越线即标该配置资源不合格，不沿用 $0.132$ h/run。6 次正式 PHNN 可先按 3–8 CPU h 作情景预算；全套含调试暂估 100–180 h、另留 4–6 周异构实现，均为规划范围，实测仍为 unknown。NRI 图推断和两阶段预处理同样计费。

$N=32$ 的 `GraphNeuralODE` 可把 `_readout(reference)` 在一次 rollout 内计算一次，与各次动态读出相减并复用于端口与 RHS。训练时保留计算图、不 `detach`，不能跨参数更新、rollout、参考、设备或 dtype 复用。入表前比较逐步输出、损失以及全部参数梯度，记录实际耗时/RSS；这里仅裁定数学等价优化。

## 9　m1、m2 · 下一位实施者的逐处同步口径

**m1。** `val_trajectory_audit.md` 文首、§2.2 两处及两份简报改为：“在 12 条可初始化失稳事件中，模型 0/12 触发失步，7/12 在 8 s 尾窗严格达稳。其余 5 条涵盖最快 4 条（0.14、0.19、0.21、0.25 s）和第 7 快的 `pilot_0239`（0.42 s，32 帧）；并列 0.33 s 的 `expansion_0186`、`pilot_0152` 占第 5、6 位，均达稳。”切勿把尾窗判据 $\max_i|\omega_i-\omega_{\rm COI}|<.005$ pu 写成共模速度本身小；达稳并不识别轨迹最终吸引子。历史判断仅据冻结表，本次没有重放轨迹。

**m2。** 两简报引用关系写成“正文一致镜像（相对链接按所在目录调整，换行编码可不同）”，保留各自实际哈希与原记录。当前简报、镜像、Campaign 和方法表随后同步：① 奇对称核与振子 $\kappa=.8$，WC 以 Dale 符号与有符号调制幅值为准；② WC 已选定，含电流接口、四块稀疏图和线性 $u$ 消融；③ 114 个新窗、$w,K_{\rm aug}$、节点干扰与列归一化/四折/5%/NNLS；④ 冻结主干与 [.50,2.20] 重标盒；⑤ 双层 LUPI、两种无特权与重写的 G3；⑥ 新噪声和预算、已完成本次 Scientist 审定、v3.1 独立复核与实际运行仍待完成。去掉旧“待 Scientist 二选一”、旧频段/盒和“字节一致”。历史的 HTTP 400 与暂定作者回执不追改。

简报原有六项范围护栏照录其原意：等价类贝叶斯/Laplace 后验另议；不另建商空间/集合估计算法；跨域跨规模只检验模板重拟合；不另建 Datasets & Benchmarks 或 RoA 求解器；实验矩阵维持十字星；新高阶 FOM 为 0、IEEE-39 test 关闭。代码决定为**交付规格供同步**，模型决定为**无晋升**，科学决定为**采用有限条件定理及两套静态定标，动态效应待辨别**。

## 附　静态复算摘录

以下可放入同一 Python 3.12 进程，使用 NumPy/SciPy。第一段对每个尺寸输出全部正自然频率和两种开放格的商空间谱指标。第二段用复步雅可比核对 WC 的全维根；仅评价代数函数，从不推进时间。

~~~python
import numpy as np

for n in (8, 16, 32):
    def stream(k):
        return np.random.default_rng(np.random.SeedSequence([20261128, n, k]))

    ring = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    pool = [(i, j) for i in range(n) for j in range(i + 1, n)
            if (i, j) not in ring and (i, j) != (0, 2)]
    picks = stream(0).choice(len(pool), n // 2 - 1, replace=False)
    edges = sorted(ring | {(0, 2)} | {pool[k] for k in picks})
    B = np.zeros((n, len(edges)))
    for k, (i, j) in enumerate(edges):
        B[i, k], B[j, k] = 1, -1
    C = np.abs(B)
    m = stream(1).uniform(.7, 1.4, n)
    d = stream(2).uniform(.12, .25, n)
    c = stream(3).uniform(1.2, 2.4, len(edges))
    ref = stream(4).uniform(-.35, .35, n)
    eta = B.T @ (ref - ref.mean())
    K = (B * (c * np.cos(eta))) @ B.T
    scale = np.linalg.eigvalsh(K / np.sqrt(np.outer(m, m)))[1] / (2*np.pi*.8)**2
    mass = scale * m
    D = np.diag(np.sqrt(scale) * d)
    freqs = np.sqrt(np.linalg.eigvalsh(K / np.sqrt(np.outer(mass, mass)))[1:]) / (2*np.pi)
    assert len(freqs) == n-1 and np.all((.44 < freqs) & (freqs < 5))
    beta = np.zeros(len(edges))
    ix = stream(5).choice(len(edges), n, replace=False)
    beta[ix] = stream(6).uniform(.65, .8, n) * c[ix]
    loss = stream(7).uniform(.25, .50, len(edges)) * c
    I = np.eye(n)
    O = np.zeros_like(I)
    R = np.diag(1 / mass)
    H = (B * (beta * np.sin(eta))) @ C.T + .35*I
    print(n, "natural Hz", freqs)
    for sym in (False, True):
        Ks = K - (C * (sym * loss * np.sin(eta))) @ B.T
        J = np.block([[O, I, O, O, O], [-R@Ks, -R@D, R, -R@H, -1.8*R],
                      [O, -.175*I, -.3375*I, O, O], [O, .2875*I, O, -.225*I, O],
                      [O, .2375*I, O, .175*I, -.1325*I]])
        eig = np.linalg.eigvals(J)
        assert np.count_nonzero(np.abs(eig) < 1e-8) == 1
        eig = eig[np.abs(eig) >= 1e-8]
        fast = np.abs(eig.imag) > 1
        ratio = np.min(np.abs(eig.imag[fast])) / np.max(np.abs(eig[~fast]))
        assert eig.real.max() < 0 and ratio >= 8
        print(n, sym, "spectral abscissa", eig.real.max(), "separation", ratio)
~~~

~~~python
import numpy as np
from scipy.optimize import root

for n in (8, 16):
    ne, ni = 3*n//4, n//4
    W = np.zeros((n, n))
    for i in range(ne):
        W[i, (i-1) % ne] = W[i, (i-2) % ne] = 1.25
        W[i, ne+i//3] = -1.5
    for k in range(ni):
        W[ne+k, 3*k:3*k+3] = 2/3
        W[ne+k, ne+(k-1) % ni] = -.5
    C = .05*W
    C[:ne, :ne] = .8*W[:ne, :ne]
    C[:, ne:] = -.02*W[:, ne:]
    bias = .5 + np.log(.15/(1-1.15*.15))/5

    def field(z):
        r, u, a, m = np.split(z, 4)
        d = r-.15
        K = W + C*np.tanh(a[:, None] + a[None, :])
        current = bias + u + K@d - 1.5*m + .01*np.tanh(a)
        sigmoid = 1/(1+np.exp(-5*(current-.5)))
        return np.r_[(-r+(1-.15*r)*sigmoid)/.08, -.6*u-.02*d,
                     -.5*a+2*d-.1*np.tanh(a), -.3*m+.02*d+.01*np.tanh(a)]

    def jac(z):
        return np.array([field(z+1j*1e-20*e).imag/1e-20 for e in np.eye(4*n)]).T

    for rate in (.15, .20, .85):
        d = rate-.15
        start = np.repeat([rate, -d/30, 4*d, .1*d], n)
        sol = root(field, start, jac=jac, tol=1e-11)
        residual = np.max(np.abs(field(sol.x)))
        eig = np.linalg.eigvals(jac(sol.x))
        assert sol.success and residual < 1e-11
        if rate != .20:
            assert eig.real.max() < 0
        if rate == .15:
            mode = eig[np.argmin(np.abs(eig.imag-6.893986588))]
            assert 1 < mode.imag/(2*np.pi) < 2
        print(n, rate, sol.x[[0, ne]], residual, eig.real.max(), sum(eig.real > 0))
~~~

**审定回执。** 指定 Scientist 已审阅两份定标回执、完整方程、拓扑及 Jacobian 构造，并签发本规格的数学与设计结论。基准 1 的矩阵和基准 3 的根另在共享解释器中复算通过。本轮 ODE 轨迹积分、训练更新、新增 FOM 与 IEEE-39 test 访问均为 0。


## 10　保留的算法、基准 2 与判别门槛

本节保留 v3.0 中未被规范替换的细节。所有实测成本、资格和效应均尚待运行验证；下列配置与门槛不是已取得结果。振子环境与幅值隐态的速度反馈限于第 2 节的常对角线性项；机械快通道按第 4 节公开字典，禁止借编码器或端口监督向部署提供未来信息。幅值与环境方程中的邻边输入只进入隐态导数，不作为端口瞬时直通。严格冻结及重标规则只采用第 4 节的 `[.50,2.20]` 盒，旧 `[.75,1.40]` 盒与“必含真值”论证均已撤销。

### 10.1 Stage 2、初始化与监督

1. **精确 Stage 2 方程（基准 1/2 振子/摆阵载体，$5N$ 维）**：
   $$\dot\theta = \omega, \qquad M\dot\omega = \hat\tau_m - \tau_e - D\omega \quad (\text{摆阵加已知 } -G\sin\theta),$$
   $$\tau_e = B\big[(c + \beta\odot\tanh(s))\odot\sin\varphi - c\odot\sin\eta\big] + |B|\big[g\odot(\cos\varphi-\cos\eta)\big] + S_e e + \alpha_1\odot\tanh(a) + \alpha_2\odot\tanh^2(a),$$
   - 机械端口采用 $\hat\tau_m=S_p p-b\odot T(\omega)$，$T$ 与可学习 $b$、各方法可用信息完全采用第 4 节；不能省略该快通道后宣称同构公平。机械隐态保留原受控字典，环境/幅值假设不扩大到机械端口。
   - **严禁 `env_edges`**：相对旧 `src/kiki/models/osgkd.py` 第 230 行，**彻底删除**环境端口上的任意无约束边特征直通项 `env_edges`，硬性落实 $\mathcal{F}_{\mathrm{env}}$。
   - **受控残差隐态方程（不将机械态固定为常数）**：以下 $\tilde\omega_i=\omega_i/\omega_{\mathrm{scale},i}$，尺度从训练冻结；保留共同速度，不减逐帧均速。
     $$\dot p_i = -\lambda_{p,i} p_i + w_{p\omega,i}\tilde\omega_i + w_{pt,i}\tanh(p_i) + w_{p\omega t,i}\tanh(\tilde\omega_i),$$
     $$\dot a_i = -\lambda_{a,i} a_i + w_{a\omega,i}\tilde\omega_i + w_{at,i}\tanh(a_i) + \sum_{e\in\hat S_c(i)}\big(v_{a,e,i}^{(1)}(\sin\varphi_e-\sin\eta_e) + v_{a,e,i}^{(2)}(\cos\varphi_e-\cos\eta_e)\big),$$
     $$\dot e_i = -\lambda_{e,i} e_i + w_{ea,i} a_i + w_{e\omega,i}\tilde\omega_i + w_{et,i}\tanh(e_i) + w_{eat,i}\tanh(a_i) + \sum_{e\in\hat S_c(i)}\big(v_{e,e,i}^{(1)}(\sin\varphi_e-\sin\eta_e) + v_{e,e,i}^{(2)}(\cos\varphi_e-\cos\eta_e)\big),$$
     所有衰减率均通过 $\lambda_{\cdot,i} = \operatorname{softplus}(\tilde\lambda_{\cdot,i}) + 0.05\ \mathrm{s}^{-1}$ 参数化，提供严格正衰减率；常轨线仍须按数据与实际右端判断，不能由正衰减率断言轨线导数处处非零；且 $\dot a,\dot e$ 的邻边输入严格掩码至冻结骨架 $\hat S_c$。

2. **等变图短历史编码器 $I_\phi$ 与硬性零 $\mathcal{O}_1$ 泄漏白名单断言**：
   - 架构沿用已验证的 `OSGKDGraph._GraphEncoder`（`src/kiki/models/osgkd_graph.py`，`hidden_size=20`，节点参数共享、参考历史扣减零点对齐，严格保留末点 $(\theta_{\mathrm{obs}}(0),\omega_{\mathrm{obs}}(0))$，原实现 **1,928 可训练参数**；新实现与 v2 和 `GraphNeuralODE` 的编码器容量须实测核对）。
   - **静态 + 运行时零泄漏白名单断言（Run 0 强制门禁）**：
     1. 接口层强制 `initialize(packet)` 仅接受 `MotionPacket`，校验顶层键集合**精确等于** `{"history", "constants"}`，且 `history` 键严格限定为 `{"t", "theta", "omega"}`（基准 3 为 `{"t", "r"}`），出现任何特权键（`targets`, `tau_m`, `tau_e`, `I_drive`, `I_net`, `truth`, `label`, `p`, `a`, `e`, `u`, `m`）立即抛 `PermissionError`；
     2. 静态 AST 检查验证 `initialize()` 与 `rollout()` 源码及调用链零引用特权字段名；
     3. 运行时污染单测（Taint Test）：固定合法运动包，在验证副本中将聚合端口、神谕字段与禁止使用的 metadata 分别覆写为 `NaN` 与 `+1e9` 两份副本，断言 `initialize` 与 `rollout` 输出与原事件**逐比特完全一致（`atol=0, rtol=0`）**。

3. **损失权重、课程与优化器（显式规避 `frozen_p` 3× 收缩陷阱与 `P2a` 历史精修灾难）**：
   - **总目标**（$\mathcal{O}_1$ 仅作训练监督目标，绝不进入输入）：
     $$\mathcal{L}_{\mathrm{Stage2}} = \mathcal{L}_{\mathrm{motion}} + 0.5\,\mathcal{L}_{\mathrm{port}} + 0.5\,\mathcal{L}_{\mathrm{port},0} + 0.05\,\mathcal{L}_{\mathrm{cons}} + 10^{-4}\|\beta\|_1,$$
     - $\mathcal{L}_{\mathrm{motion}}$：归一化 $(\theta,\omega)$ 在有效未来区间 $(0, T_s]$ 的均方误差（失稳轨线在截止点截断，不填充）；
     - $\mathcal{L}_{\mathrm{port}}$：归一化 $(\tau_m, \tau_e)$ 在 $(0, T_s]$ 的均方误差（权重 `0.5`，强于 v2 的 `0.1`，锚定开放残差归属）；
     - **$\mathcal{L}_{\mathrm{port},0}$（反 `frozen_p` 3× 收缩专设项）**：在部署起点 $t=0$ 直接以 $\mathcal{O}_1$ 监督 $\hat\tau_m(0)=S_p p_0-b\odot T(\omega_0)$ 与 $\hat\tau_e(0)$（权重 `0.5`），配合可训练阻尼 $\dot p$，约束编码器压缩 $\|p_0\|_2$ 的跨模块代偿；是否改善由匹配对照验证；
     - $\mathcal{L}_{\mathrm{cons}}$：推进 $0.1\ \mathrm{s}$ 后隐态与编码器对未来 $0.1\ \mathrm{s}$ 运动窗重新编码的一致性损失（权重 `0.05`）；
     - **永久禁用 `P2a` 历史一步精修**：无论训练期或部署期均不做初态梯度精修，杜绝过拟合 11 帧噪声导致未来风险恶化 44× 的灾难。
   - **三段课程与优化器**：Phase A（$T_s=1\ \mathrm{s}$, 64 步）$\to$ Phase B（$T_s=2\ \mathrm{s}$, 64 步）$\to$ Phase C（$T_s=4\ \mathrm{s}$, 128 步），总计 **`256` 次接受更新**；物理与有效批量均为 `4`（按训练运动可得结局和训练子折历史能量四分位配平，来源配对不跨折）；Adam 分组学习率（编码器 `3e-3`、残差 ODE `1e-2`），8 步线性预热 + 余弦退火至 `0.1×`，梯度范数裁剪 `5.0`，每步后硬投影约束与支持掩码；训练内部 RK4 步长 `0.01 s`（训练前强制过 `0.01 vs 0.005 s` 双网格梯度资格），评分强制过 `0.005 vs 0.0025 s` 双网格资格。

---


编码器参数数目为原实现口径；新实现须实测复核。训练一致性项只在训练期从合法运动子窗计算；部署不得重新编码、初态梯度精修或逐事件拟合。所有归一化按第 6 节只从相应训练折确定。聚合端口与真分项评分严格分开。

### 10.2 基准 2 与十字星组织

| 切片轴 | 格点设置 | 方法清单 | 种子与独立单元 |
|---|---|---|---|
| **基准 2（升级摆阵）** | $N=8$，强开放/强调制 + 已知重力 $-G\sin\theta$，每系统 `96/32/128`（train/val/独立合成确认集）（含 64 翻转/失稳），噪声 `0.01` | 核心 6 方法（`PDD-Frozen`, `Stage-1-Only`, `End-to-End v2`, `No-Privilege Same-Arch`, `Graph Neural ODE`, `PHNN/LGNN`） | **3 个独立系统种子**（`31, 41, 51`）各 1 优化种子（3 独立物理系统） |

基准 2 的已知重力项在预测和纯运动弱式中使用同一符号，不能由环境项补偿。只复用振子的冻结/重标、部署白名单和数值资格，不把参数或性能跨系统等同。相同模板在三独立物理系统重新拟合，分别报告各系统结果。

十字星仍只做中心锚格、工况递进、全轨迹噪声、规模、基准 2、基准 3 与理论负对照，不展开全笛卡尔积。中心锚格种子为 `24121,24122,24123`；规模轴 N=16 两优化种子、N=32 一优化种子。`static_closed`、`strong_mod_closed` 各两优化种子，`strong_open_sym` 三优化种子；噪声两侧各两种子。新 Two-Stage 对照及去重后的数量、课程、Run 0 项目以第 8 节的 161 次 ODE 训练、41,344 次课程更新为唯一总账；不沿用旧表的 128 次或 40,576 次口径。固定图上的优化种子不当作独立物理系统。

将来独立合成确认集拟为每格 128 条（64 稳/64 失稳），train/val 拟为 96/32 条；标签配额、历史可初始化、完整窗覆盖和分界激励须实际核验。四类激励为 `distributed_strong`、`local_fault`、`latent_overload`、`boundary_near_separatrix`；沿扰动射线分界包夹的来源簇在划分前分组，不能把同源近邻分到训练与确认两侧。确认集在生成器冻结后锁定，Run 0 不用它改设计。所有“test”命名仅指未来独立合成确认，IEEE-39 test 始终关闭；本次没有读取或生成这些数据。不预言固定样本量必达指定检验力。

### 10.3 G1、G2、G4 原登记效应量与统计单位

G3 完整采用第 1 节的匹配 Two-Stage 主比较。下表用于将来三优化种子与独立合成确认，Run 0 只在合成 train/val 作冒烟。VPT 是归一化瞬时状态误差首次超过 .20 前的有效时长，上限 4 s；尺度从训练冻结。振子失稳连续评分为自治轨线 `max ptp(theta)`，无额外分类头；WC 用第 5 节对应阈值与连续发放率评分。

| 门禁 | 核心比较对 | 预登记定量通过阈值（须**全部同时满足**） | 统计检验与区间方法 | 失败 / 止损触发条件 |
| --- | --- | --- | --- | --- |
| **G1（Stage 2 必要性）** | `PDD-Frozen` vs `Stage-1-Only` | ① 3 种子均值 `MSE@4s` 相对降低 **$\ge 40\%$**（比值 $\le 0.60$），且 **3/3 种子同号**；<br>② 3 种子均值 `VPT` 净提升 **$\ge +0.60\ \mathrm{s}$**（3/3 种子同号）；<br>③ 失稳召回净提升 **$\ge +15\ \mathrm{pp}$**（同时稳定误报率 $\le 10\%$）**或** 失稳 `AUROC` 提升 **$\ge +0.08$**。 | 逐事件配对自助法（Paired Event Bootstrap, $B=10{,}000$）计算 95% CI（严格不跨零）；召回采用 **McNemar 精确配对检验**（$p<0.01$）；AUROC 采用 **DeLong 配对检验**（$p<0.01$）。 | **Run 0 早停止损**：若 `Stage-1-Only` 的 `MSE@4s < 0.05` 或 PDD 相对降幅 $<25\%$，暂停全矩阵，先核查生成器调制/开放 RMS 占比是否落入目标区间及 Stage 2 是否欠拟合。 |
| **G2（解耦优于端到端）** | `PDD-Frozen` vs `End-to-End OSGKD v2` | ① 静态边 `AUPRC` 提升 **$\ge +0.15$**，静态边 `F1 >= 0.90` 且高出 v2 **$\ge +0.15$**（3/3 种子）；<br>② $\mathcal{O}_2$ 内部/环境分离 `Port Sep RMSE` 相对降低 **$\ge 30\%$**（3/3 种子）；<br>③ 预测/失稳非退化护栏：`MSE@4s` 不劣于 v2 超过 $10\%$ 且失稳 `AUROC` 不劣于 v2 超过 $0.03$。 | 配对 Bootstrap 95% CI（不跨零）；`Port Sep RMSE` 配对 **Wilcoxon 符号秩检验**（$p<0.01$）。并同步核查 `Bounded-Recal`：仅当其跨 3/3 种子降 `MSE@4s` $\ge 10\%$ 且 $\hat c$ 相对误差恶化 $\le 5\%$ 时才记为有效补偿消融，否则坚持 `PDD-Frozen` 主干。 | 若 v2 在长时预测与失稳上全面碾压（PDD 退化 $>15\%$）或静态边 F1 优势 $<0.05$，暂不支持该预算下硬冻结解耦主干优势，转查 Stage 1 残差偏差传递。 |
| **G4（定性失稳分岔保真）** | `PDD-Frozen` vs `Graph Neural ODE` & `Latent ODE`（并列 `Direct Classifier`） | ① `PDD-Frozen` 失稳召回 **$\ge 70\%$**、稳定误报 **$\le 12\%$**，且召回高于两黑箱 ODE 至少 **$\ge +20\ \mathrm{pp}$**；<br>② 失稳 `AUROC >= 0.85` 且高于两黑箱 ODE **$\ge +0.08$**；<br>③ 稳定子集 `MSE@4s` 不超 `Graph Neural ODE` 的 $1.5\times$；<br>④ 诚实并列 `Direct Classifier`（不强求反超有监督分类器，展示在产出连续轨迹+物理图前提下的可比性）。 | 64 条失稳 **McNemar 精确检验**（$p<0.005$）；128 条 **DeLong 检验**（$p<0.005$）；精确 **Clopper–Pearson 95% CI**。并在基准 3（Wilson–Cowan 64+64 癫痫跃迁）复验跨方程族一致性。 | 若黑箱 ODE 在大样本合成失稳上亦保持高召回（$>65\%$ 且与 PDD 差 $<10\ \mathrm{pp}$），收缩当前合成数据上的 C3/G4 主张；IEEE-39 或分布外优势须另行登记并取得相应证据，不由本阴性结果推出。 |

配对自助法以事件或来源事件簇为单位；事件簇中相关样本不能当独立事件。F1/AUPRC 按物理图与优化种子直接列值，不对同一图的边重采样制造跨系统置信度。表中 Bootstrap/Wilcoxon 只用于有合理配对单位的事件误差；不把三种子当作足以执行 p<.01 的 Wilcoxon 样本。召回的配对二元差异用同一失稳事件，AUROC 用同一合法确认事件；相关簇需按簇处理。三项分别报告，全部满足才作联合主张。效应阈值与统计不确定性分别判断，不以未显著证明等效。

### 10.4 Run 0 其余保留门槛与前瞻适配

基准 1 锚格单种子 `24121`：Stage 1 F1≥.90、AUPRC≥.95；`Stage-1-Only` MSE@4s≥.10 或失稳召回≤65%；PDD 相对其 MSE 降≥35%、VPT 增≥.50 s；PDD 静态边 F1 比 v2 高≥.15。基准 3：Stage 1 有向 AUPRC≥.90，PDD 相对 Stage-1-Only 的 MSE 降≥30%、跃迁召回≥65%。全部零泄漏资格、训练/评分双网格资格及 N=32 峰值 RSS≤4 GiB 均须实际通过。预算、九项端点和 21 个超阈值采样点的计时规则按第 8 节。

若 Stage-1-Only MSE<.05 或 PDD 降幅<25%，暂停矩阵扩张，诊断 RMS、覆盖和优化充分性；不能由低误差自动断言机制不存在。若 Stage 1 F1<.85，按第 4 节依次查窗、谱、秩/夹角、噪声和投影残差，不删除 .5 Hz 窗。若冻结主干比重标或 v2 差≥15%，报告投影目标偏移、越盒/触边、优化与端口误差；不能从单次冒烟自动晋升重标主干。黑箱召回高时收缩相应合成主张，不自动转移为 IEEE-39 优势。任何工况、课程或算法适配需形成新冻结版本、保留父子谱系，且不得回看合成确认集改造生成器。

原方案暂不展开等价类 Bayesian/Laplace、自动方程空间搜索或混合切换、Datasets & Benchmarks 独立路线或 RoA 扩展。新增高阶 FOM=0，TASK4 暂停，IEEE-39 test 关闭。代码决定为文档合同修订，模型决定为无晋升；科学决定限于本规范的有条件命题及静态定标，轨迹效应待检验。
