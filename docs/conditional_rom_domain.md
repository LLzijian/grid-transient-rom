# 固定参数物理ROM的一个解析收敛内域

这条充分命题把模型的耗散结构进一步连接到一个可计算的吸引域内估计。它适用于下述固定参数、无输入的连续时间ROM；实际应用时须核验具体参数和初态。本文不报告任何已获证的模型或初态，也不替代项目完整L2要求。

## 系统与条件

考虑10个节点。每条边的标准关联向量为 `b_e`，满足 `b_e^T 1=0`、`||b_e||=sqrt(2)`。令

\[
P=I-\mathbf1\mathbf1^T/10,\qquad q=P\delta\in S=\mathbf1^\perp.
\]

只消去共同角度，保留全部十维转速。商空间状态 `(q,omega,p,r)` 的维数为 `9+10+10+10=39`，方程为

\[
\begin{aligned}
\dot q&=P\omega,\\
M\dot\omega&=-\nabla U(q)-D_\omega\omega+GS_p p-r,\\
\dot p&=-G\omega-R_pS_p p,\\
\dot r&=\kappa\omega-Q(q)r,\\
U(q)&=\sum_e k_e[\cos\phi_e-\cos(\phi_e+b_e^Tq)-\sin\phi_e\,b_e^Tq].
\end{aligned}
\]

所有参数固定，采用同一个工作点相位展开约定，且满足：

- `M,S_p,R_p` 为正定对角矩阵，`D_omega` 为半正定对角矩阵，`G` 为实对角矩阵；`kappa>0`、`lambda>0`。
- 全部 `k_e>=0`。正势能边集 `E_U={e:k_e>0}` 构成连通图，且这些边满足 `|phi_e|<pi/2`。
- `Q(q)` 对称、局部Lipschitz，且 `Q(q)>=lambda I`。没有额外的 `p-r` 交换项。

例如，固定非负记忆边权 `a_e` 和门函数 `g_e(q)=(1+cos(b_e^Tq))/2` 可给出 `Q=lambda I+sum_e a_e g_e b_e b_e^T`。记忆边与势能边承担不同作用；相位条件只约束正势能边。固定参数的全零记忆图可以定义为 `Q=lambda I`，这一数学定义不提供参数训练在零图处的梯度。

## 内域公式

只对 `e in E_U` 定义

\[
\begin{aligned}
\gamma_e&=(\pi/2-|\phi_e|)/2,\\
D&=\{q\in S:|b_e^Tq|<\gamma_e,\ e\in E_U\},\\
w_e&=k_e\cos(|\phi_e|+\gamma_e),\\
L_w&=\sum_{e\in E_U}w_e b_e b_e^T,\\
\mu&=\lambda_2(L_w)>0,\qquad \gamma_{\min}=\min_{e\in E_U}\gamma_e,\\
\rho&=\mu\gamma_{\min}^2/8.
\end{aligned}
\]

这里的谱隙 `mu` 对应欧氏零和角坐标 `S`。定义ROM储能函数

\[
H=\tfrac12\omega^TM\omega+\tfrac12p^TS_pp+\frac{\|r\|^2}{2\kappa}+U(q).
\]

**命题。** 集合

\[
K=\{(q,\omega,p,r):q\in\overline D,\ H\le\rho\}
\]

是紧致正不变集，且实际包含于 `q in D`。从 `K` 出发的连续解前向完备，并趋于商空间原点 `(0,0,0,0)`。该平衡点局部渐近稳定，`K` 给出其吸引域的一个保守内估计。初态必须同时满足 `q_0 in D` 和 `H_0<=rho`。

## 证明

在 `closure D` 中，每条正势能边满足

\[
|\phi_e+b_e^Tq|\le|\phi_e|+\gamma_e<\pi/2,
\qquad \nabla^2 U(q)\succeq L_w.
\]

`D` 凸且包含0，`U(0)=0`、`grad U(0)=0`。沿0到 `q` 的线段积分，在 `S` 上得到

\[
U(q)\ge\mu\|q\|^2/2.
\]

正势能图连通，有限的边角差约束和零和条件使 `closure D` 有界。其相对边界至少有一条正边饱和；由 `||b_e||=sqrt(2)` 得

\[
\|q\|\ge\gamma_{\min}/\sqrt2,\qquad
U(q)\ge\mu\gamma_{\min}^2/4=2\rho.
\]

因此 `rho` **等于上述边界下界的一半**。`H` 的各二次项及势能下界使 `K` 闭且有界，故紧；边界能量至少为 `2rho>rho`，又排除了 `q` 到达 `D` 边界的可能。

由于 `grad U` 属于 `S`，有 `grad U^T P omega=grad U^T omega`。按方程求导，势能、机械端口及隐力端口的功率逐项抵消：

\[
\dot H=-\omega^TD_\omega\omega-p^TS_pR_pS_pp
-\frac{r^TQ(q)r}{\kappa}\le0.
\]

`H` 不含 `Q`，因此此处没有 `dot Q` 项。能量不增及边界严格分离使解留在 `K`；局部Lipschitz向量场在这个紧集的邻域保证解可延拓到全部 `t>=0`。

零耗散集合必有 `p=r=0`。若一条轨迹始终位于其中，则 `r` 恒为0，由 `dot r=kappa omega` 推出全部十维 `omega=0`，再由转速方程推出 `grad U(q)=0`。`U` 在凸集 `D` 的零和坐标上严格凸，唯一驻点为0。因此零耗散集合的最大不变集是商空间原点。LaSalle原理给出吸引性；正定储能函数同时给出局部Lyapunov稳定性，命题成立。

## 应用范围与一次核验所需信息

应用这条命题时，应固定同一模型导出、坐标和方程，核验全部符号、严格正条件、正势能图连通性及相位裕量，再得到可靠的 `mu` 下界并构造保守 `rho`。普通浮点特征值在临界情况下不足以充当严格下界。若使用 `0<mu_lower<=mu`，可在公式和证明中一致采用 `mu_lower`。

初态检查须同时包含相位域与能量条件。全局势能非凸，远处分支的低能量不能仅凭 `H<=rho` 获证。估计得到的ROM初态只能直接获得关于该确定初态的结论；覆盖初态误差集合需要另有可信的集合界。

结论是连续时间ROM的渐近收敛。它没有给出指数速率、固定步RK4逐点不越界、暂态LOS阈值从不越界或真实高阶物理系统的安全保证。原40维系统中共同角度已被商掉，不能据此称原空间仅有一个平衡点或已证明共同角度极限。`H` 是ROM内部储能，`kappa` 保持等效刚度类单位，不将其称为惯量或将 `H` 直接称为真实SI能量。

本命题的条件不满足，表示该充分证明不能应用；这本身不是模型失稳的判定。

## 推导与审查记录

2026-10-29由实际 `gpt-6-astra` 推导，实际 `gpt-6.1-sol` 作独立符号挑战，root采纳上述条件命题。原推导中关于“一半”的措辞已在本稿修正，数值常数不变。推导和独立挑战均未读取实际端点参数或事件，也未进行端点认证、训练或覆盖率计算。原始记录保存在研究归档分支 `codex/force-memory-20261028` 的 `reports/astra6_conditional_rom_domain_20261029_raw.md` 和 `reports/conditional_rom_domain_20261029_review.md`。
