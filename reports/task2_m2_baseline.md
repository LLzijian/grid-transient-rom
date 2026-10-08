# TASK2 Milestone 0：M2 clean baseline（Loop 0019）

2026-09-08，起点`436b721`。问题：建立统一、可复现的最小二阶比较基准，
不是检验它能否达到最终精度；完成后冻结，不再增加M2结构。

固定：10机同步旋转坐标，角度相对唯一基准平衡角，速度为原始绝对pu偏差×
omega_s；保留共同频率信息供后续governor使用。报告同时使用COI角度/速度误差。
`mass=M_system/omega_s`，`delta_dot=omega`，
`mass*omega_dot=(Pm_clear-Pm_star)-B*K*(sin(B.T*(delta+delta_star))-sin(B.T*delta_star))-D*omega`。
K、D非负；M2对决策时刻Pm作ZOH。未把端口Pe当成电磁转矩，也不使用未来Pm。

唯一development split：train227/val74；test60不载入。统一在清除后50ms开始，
真值在该时刻已LOS的事件记作决策前事件，不进入prospective评分。
训练用全部其余训练事件的真实前缀，最长决策后8s，不补齐。
弱积分宽度1/5/20点、每宽度最多40个分散窗口；事件等权，固定ridge1e-6，
非负最小二乘一次拟合，不调种子、ridge或候选库。

评价：0.2/0.5/1/2/5/8s完整可观察人群分别计数，partial prefix另报；
每个事件先计算COI delta(rad)、omega(rad/s) RMSE，再用训练集等事件RMS尺度
计算无量纲联合RMSE；报告pooled/mean/worst-10%及各母线组最差表现。
对全部决策时刻仍存活的验证初值滚动固定8s，按冻结LOS规则评估分类，不用真实
轨迹终止长度决定预测时域。时间与质量单位、无未来输入、无测试读取为硬门。

基线验收门：参数有限非负、平衡点精确、合成参数恢复、RK4与独立DOP853
一致、所有验证结果/数值失败明确计数；不以M2精度好坏阻止建立比较基准。
后续M3-Pm以真实训练Pm辨识最低阶ODE，比较同一电气骨架上的ZOH与动态Pm；
准确记录机械侧解释能力，收益有限也按TASK2进入最小M4研究，不提前造能量框架。
