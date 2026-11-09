# 共享常数范数必要条件：冻结前协议草案

2026-10-31。Run候选 `p_constant_norm_20261031`，同属Campaign `force_memory_closure_20261030`；当前仅设计与实现审查准备，未登记科学执行、未释放Job。
本协议和Scientist设计共同约束后续执行；只登记发布数值条件问题，不改变模型、残差系数或已有判定。

## 问题与精确范围

固定父θ，问已发布范数必要条件是否还容许任意每机共享实常数z。
对归一权重范数有 `|R−|z|| <= ||r−z||`，因此相容所需 `||r−z||<=T` 的必要条件是
`|z| ∈ [max(0,R−T),R+T]`。每机36条必要区间，L为全部下界最大值，U为全部上界最小值。
L>U才为空，L=U非空；非空不证明任何真实z可行，也不识别符号、常值驱动、有效阶数或合法初始化。

本轮R和ν专指父公开JSON中binary64值的精确有理数R_hat/ν_hat，T_hat为整数10精确乘ν_hat。
原聚合/平方根误差包络unknown；不由一ULP、旧1e−12门或经验容差认证原有限残差范数。
精确10乘法与父binary64阈值乘法不同；本轮不重判父类别，也不声称同一浮点门的全常数严格否证。
空交集的结果句必须带“以已发布binary64范数/ν为精确输入”条件；不外推有限数据或连续物理系统。
这是便宜的解释筛查，不是自主预测收益或新机制发现。

## 来源、覆盖与资格

以source_bindings中八个精确路径的bytes/SHA为唯一入口；父commit为1fcfd53。
唯一数值来源是父analysis_summary SHA79150ea05924fd38348972f87ce5002b786abd5ec249ddb814d45481a1ef447f。
只取overall60/stage120的R_Q_before、R_L_before、原ν及标识/覆盖；不用校正后R、不读η/θ/scale。
其他公开文件只做定义与身份；父私有归档控制只核原字节SHA，不解析任何私有科学载荷或跟随其中路径。

固定整体60格=十机×all/stable/unstable×steps4/8，阶段120格=十机×stable/unstable×history/cross_t0/future×steps4/8。
每格Q/L各一条，共360条、十个交集；每机36条。全纳入一次，不另作未来子集交集或择格。
原分母N_original始终为all297/stable267/unstable30，所有阶段均不改原分母；它与N_available分开。
整体N_available=原分母；阶段stable所有相位/窗均267，unstable的history两窗30、cross_t0两窗27、future的steps4为27/steps8为22。
这些是父冻结公开表的覆盖元数据，不改变权重、补样本或删格；原先“阶段历史/跨界保留原N”专指原分母，现明确实际有窗数以免误读。
整体μ=1；阶段0<μ<=1、正整数windows、0<N_available<=N_original。无窗/缺键/重复键/非有限/负R或ν、范围或身份不符整次停止unknown，不能丢格。
阶段权重仍w=1/(N_group*m_e_all)，条件范数除μ归一；不重新让可用事件等权。
单位R、ν、T、z、L/U为s⁻²，C为s⁻⁴、μ无量纲。11点100ms部署窗口及未评分尾段不变。

## 两路径解析与精确运算

第一路径正常JSON数值解析为binary64，再用as_integer_ratio取得精确分数；第二路径保留原JSON数值token，精确解析十进制有理数。
对R_Q_before/R_L_before/ν以及μ验证最近偶数舍入：用binary64前后相邻数的精确中点及significand最低位奇偶判断；ULP仅校验token到binary64身份。
验证所有数值token映射后才算任何科学区间；JSON重复键、NaN/Infinity拒绝。整数键与分母数量须严格整数而非bool。
不从表格显示位反推输入，不以显示小数参与判定。冻结支持域为0<=binary64值<最大有限binary64；最大有限数及其无法用两侧有限中点覆盖的情况停止unknown，不近似处理。
0（含±0作为同一有理数0）、最小/其他subnormal和正常数均用精确相邻数中点核验；tie包含当且仅当候选significand最低位为偶数。
只有JSON到精确有理数的身份获核，不声称保留有理数零的符号位；边界算法用确定性解析小例验证，不生成FOM/ROM。

主路径用精确有理数加减、10乘、max/min和比较。证人固定排序为tuple：(machine整数,kind_rank,group字符串,steps整数,phase字符串,representation_rank)。
kind_rank整体overall=0、阶段stage=1；整体phase固定空字符串，阶段为cross_t0/future/history原字符串按字典序；representation_rank为Q=0、L=1。
相等端点取排序首项。独立整数路径对证人端点及全部36条的排序/包含比较交叉相乘核验，不引入第二候选或重估计。
输出全部360约束及十个交集的分子/分母字符串、显示值、空/未排除、上下界证人键与核验计数；显示值仅可读，不参与判断。
证人是固定全表的证据，不是选两格重新定义实验。输入已公开，输出可公开；不反解η或私有残差。

## 有限工作、停止与决定

一个正常Job：八源身份→180格资格→两路径token身份→360约束/十交集→同目标整数证人检查。
0参数估计/QR/均值MP/新残差/聚合重算/积分/敏感界/滤波/初始化/optimizer/ROM/FOM/训练/下载/模型读写。
最多一次纯技术修复，失败/旧新源码及产物hash保留后独立复核并由root另放行；有效结果不重放，身份未知先对账。
来源/数学边界失败或非空不是修复理由，不能加容差、窗口、符号信息或额外拟合。
全部非空即停止这条必要条件支线；部分空也只报告发布数值命题，是否追求真实有限残差证书另行设计，当前不自动扩张。
当前优化健康/充分性not_applicable；来源ODE两项unknown。证据有效性限身份/解析/有理算术；机制层为发布必要条件可行性；确认none、已消费train探索。
预期代码retain_branch、模型retain_evidence、科学continue/harness observation_only；结果后root分别记录。
原val/test关闭、旧消费test永久退出选择、新FOM=0、TASK4暂停、default none/四指针/受保护范围不变。
部署仍100ms因果δ/ω与末点真p₀后共享自治ODE，不读取未来真值或事件身份。
记录实际原生wall、工具wall、CPU、token与运算计数；完整CPU/LLM/helper未知为null。关闭余额不转移。

本文件须通过独立设计和实现审查，并在执行release记录精确SHA后方可启动；没有隐含Job/训练授权。
