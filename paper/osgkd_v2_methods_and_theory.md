# OSGKD 2.0 · 方法与理论证据

从 100 ms 的运动历史推断机械作用 $p$、调制幅值 $a$ 与环境记忆 $e$。初始化器共享节点参数，通过成对相位及转速差聚合历史，并从原始读出扣除参考历史读出。随后同一开放方程在 $x=(\theta,\omega,p,a,e)$ 上积分。未来轨迹与端口按协议用于监督。真图、调制标签与隐态真值只用于冻结评分。共享编码器的形状通用性、映射后的预测与结构可辨识性由三组不同证据检验。[架构](../designs/osgkd_v2_architecture.md) · [Layer 1](../campaigns/osgkd_v2/wp2_layer1_execution_protocol.md)

## 互易边与开放功率

令 $B\in\mathbb R^{N\times |E|}$ 为定向关联矩阵，$\varphi=B^\top\theta$，$\eta=B^\top\theta_\star$。边幅值和 $s_e=a_i+a_j$，核满足

$$K_e=c_e+\beta_e\big[\exp(\tanh s_e)-1\big],\qquad 0\le\beta_e\le c_e.$$

取参考处角与幅值的一阶校正势

$$U=\sum_e\left[K_e(1-\cos\varphi_e)-c_e(1-\cos\eta_e)-c_e\sin\eta_e(\varphi_e-\eta_e)-\beta_e(1-\cos\eta_e)s_e\right].$$

因此 $F_c=\partial_\theta U=B[K\odot\sin\varphi-c\odot\sin\eta]$，且 $\mathbf 1^\top F_c=0$。对已知常值 $M\succ0$ 和 $D\succeq0$，动量方程 $M\dot\omega=\tau_m-F_c-\tau_{\rm env}-D\omega$ 与 $\dot\theta=\omega$ 给出

$$\frac{d}{dt}\left(\tfrac12\omega^\top M\omega+U\right)=
 \omega^\top\tau_m-\omega^\top\tau_{\rm env}-\omega^\top D\omega+\partial_a U\cdot\dot a.$$

等式区分互易交换、机械与环境供能、阻尼和时变耦合做功。开放供能的符号由轨迹决定。储能的全域正性与稳定证书需要额外假设。[模型实现](../src/kiki/models/osgkd.py) · [原证明与功率边界](../designs/osgkd_v2_architecture.md)

## 全局存在性

固定有限参数和冻结正定质量。由于 $e^{-1}c_e\le K_e\le e c_e$，耦合力及三角字典有界，机械输出、环境记忆和相关状态导数至多线性增长。每一项由光滑状态函数、线性读出、固定系数构成，故 $f$ 局部 Lipschitz。有限的 $A,L$ 使 $\|f(x)\|\le A+L\|x\|$，Grönwall 不等式在每个有限时间区间给出有界状态。局部唯一解因此可延拓至任意有限时刻。这一推导允许局部增长模态及长期大幅运动。[有界核实现](../src/kiki/models/osgkd.py)

## 无界核的有限时间反例

在已审查的两机储能族的不变子系统上，标量约化为 $\dot z=(\pi/2)K'(z)-z$，$z(0)=0$。对 $K=e^z$ 且 $z\ge0$，有 $\dot z\ge(\pi/2-1)e^z$，故逃逸时刻至多为 $(\pi/2-1)^{-1}$。对 $K=e^{\tanh z}$，有 $0\le z(t)\le(\pi e/2)(1-e^{-t})$。同一个构造对无界核和有界核分别积分。前者用反函数求积达到 4、8、12、16 的有限阈值及解析尾界，后者在超过前者解析上界的共同时间窗用双网格核验。结论适用该不变子系统。[反例推导](../designs/history/gftd_20261113/gftd_storage_feasibility_20261109.md) · [可复现代码](../experiments/osgkd_v2_layer1/escape_example.py)

## 经验结构选择

基准边与调制近似作用以固定转矩尺度下的训练轨迹 RMS 排序。用预登记阈值确定支撑，并从同一端点给完整和约束模型各一次相同的重拟合机会。已知真值合成试验分别衡量稀疏支撑、作用函数与轨迹预测。同步与纯幅值场景的零灵敏度以及同频观测补偿给出判断恢复失败的结构性参照。[拟合排程](../campaigns/osgkd_v2/wp2_layer1_fit_schedule.md) · [负对照](../experiments/osgkd_v2_wp2_layer1_20261127/negative_controls.py)
