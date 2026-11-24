# 零和快电气记忆 · 架构简报

**用独立快状态承接早期机间电气传递误差，并对共同转速保持直接作用为零。** 2026-11-24，用户指定原目标组第 2060 次作为本轮共同工作父点。该点在原始 `val` 的稳定 8 s 运动风险为 0.842、误报为 0/95，`train` 真实区间检出为 3/43，`val` 为 0/12（原分母 0/22）。对照保留其 50 维表示。[全量配对结果](../reports/osgkd_v2_train385_risk_alignment_20261123/result.md)

## 信息、动力学和功率

原状态 $z=(\theta,\omega,p,a,e)$ 增加一个节点向量 $f\in\mathbb R^N$。部署沿用 `motion_only_v1` 的 11 点运动历史和冻结常数。原 GRU 的末隐层连接独立线性头，再经 `tanh` 写入 $f_0$。旧初态和旧读出沿用原计算。只学习新增快闭合的系数及任务允许的原有参数。

\[
\begin{aligned}
P_0&=I-\tfrac1N\mathbf1\mathbf1^\top,\qquad F_f=P_0(Se\odot\gamma_f\odot f),\\
M\dot\omega&=\tau_m-D\omega-F_c-\tau_{\rm env}-F_f,\\
\dot f_i&=-\lambda_{f,i}f_i+\kappa_{f,i}(\omega_i-\omega_{\rm coi})/\mathrm{omega\_scale}_i
+\sum_{k\ni i}\left(W^s_{f,ik}s_k+W^c_{f,ik}c^{\rm feat}_k\right),\\
\lambda_{f,i}&=5+45\,\sigma(\ell_i),\qquad
\omega_{\rm coi}=\tfrac{\mathbf1^\top M\omega}{\mathbf1^\top M\mathbf1}.
\end{aligned}
\]

$s_k$、$c^{\rm feat}_k$ 沿用原图上相对角的参考中心化正弦、余弦。初始 $\lambda_f=10\ {\rm s}^{-1}$，$\kappa_f=0.1$，边驱动权重为零，快初态头小幅初始化，$\gamma_f=0$。原父点 $e$ 单节点自反馈的衰减率为 0.980–1.293 s⁻¹，即按原记忆方程中 $-\partial\dot e_i/\partial e_i$ 计算，保持其余状态固定。两组更新后的范围重算。快支路的齐次项到 1 s 至多保留 $e^{-5}$，实际受迫响应继续随运动演化。

投影在物理转矩坐标完成，$\mathbf1^\top F_f=0$，故共同加速度的直接贡献 $-\mathbf1^\top F_f/(\mathbf1^\top M\mathbf1)=0$。完整自主预测的共同转速仍受轨迹改变后其他端口作用影响。原储能账保留，并单列 $-\omega^\top F_f$。原质量矩阵固定正定、尺度固定正值且系数有限时，向量场由固定线性项、有界且导数有界的三角及双曲函数和其有界乘积构成，因而全局 Lipschitz。数值积分资格由双步长另行核对。

## 竞争解释与可辨识范围

上一对照的候选组用更负的环境初态减小选定机组的电气正偏差，固定 22 条在 0.03 s 的总电气误差对质量加权共同加速度的平方贡献从对照 11.75 升至候选 28.47。该关联提出独立零和闭合的假说。前序 $R_*$ 只是保存状态点上的零和校正残余下界。端口观测只给总电气转矩。增广结构的直接贡献在同一状态上通过移除 $F_f$ 读出核对，实际自主预测用训练后的原方程对照检验。

$f$、驱动、初态头和读出存在尺度与参数补偿。本轮估计可重复的总作用和预测变化，新增状态的物理身份及参数唯一性不作为采用依据。若稳定误报与长时风险受保护且电气短窗误差下降，可支持这种增广闭合在登记范围内的预测作用。若绝对误差变小但真区间检出不变，保留两类结果分别解释。现有门控与过去静态高阶势的作用范围不同，快衰减率的必要性须在后续独立对照中辨别。[前序电气分解](../reports/osgkd_v2_electrical_decomposition_20261123/result.md) · [配对协议](../campaigns/osgkd_v2/fast_zero_sum_protocol.md)
