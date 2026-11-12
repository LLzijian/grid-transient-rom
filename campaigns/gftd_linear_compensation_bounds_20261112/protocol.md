# 固定缓存凸问题：有限证书协议

状态 DRAFT，未释放。Run `gftd_linear_compensation_bounds_20261112`；parent `gftd_joint_distinguishability_20261112`。科学问题、公式及主张范围见[设计](../../designs/gftd_linear_compensation_bounds_20261112.md)。本协议只允许对冻结缓存求上下界；旧已关闭额度不转移。

## 1. 身份与读取范围

执行前绑定以下公共回执，任一缺失或不符即停止，不补算：

| 来源 | SHA256 |
|---|---|
| `results/gftd_joint_distinguishability_20261112/final_adoption.json` | `23f3dca451c5ab6083522bcc328fe4add1247e2a992712577df5935b044699ac` |
| 同 Run `science/execution.json` | `27d3ad39306c52e3fa581d48c68c54db8af2dd1da0d0fa8974c03932637a2dca` |
| 同 Run `result_audit_v2.json` | `ebc7380be6a62bebaec884d3562d026ec8d8b1901ba2b80e20b20a244d3eedd7` |
| `artifacts/gftd_joint_distinguishability_20261112/science/evidence.json.gz` | `81a64b608f7330e48eae65c19ee3cf9e9a4b2337401aaae444341a4915c14422` |

采用后执行可全字节校验最后一项并解析，只提取 `A,b,lower,upper,v`；不评价同包其他历史／轨迹字段，不重开 parent packets、原297数据或未来。矩阵、向量及证书见证留本地私有 artifacts，不进入公共报告或 Git。要求形状分别为 (5280,525)、(5280,)、(525,)、(525,)、(525,)，全为有限 binary64，逐项 lower<upper 且原 v 精确可行。A、b 按父源码的 `<f8` C序字节hash分别为 `77ec1bd31bedd68c7b468920c988e52579a781fcf900591afa19c31c54f36a44`、`8a791621faf208a8e4d714abf853869b3e4e6a6ac59d9bfe9bb17726b551fa5f`。盒和 v 的新hash作为前向记录；旧完整私有文件hash是它们的来源绑定，不冒充旧独立hash。

JSON 数值恢复 binary64 后用 `float.as_integer_ratio()` 转精确值，不先截断十进制或取显示位数。保存每个新点的原binary64字节。准备阶段绑定解释器、NumPy、SciPy、BLAS及线程、运行脚本和证书实现的源hash；不导入模型模块。继承 Python 3.12.14、NumPy 2.5.2、SciPy 1.18.0、BLAS线程1的已装环境；不安装或切换求解器。

## 2. 两阶段与证书精确计算

比较 `theta=Fraction(2048,10**12)`，等价于 `Fraction(1,2)*Fraction(64,10**6)**2`。依设计用精确整数／Fraction 点积计算 r、s、q、h、U、L。无需开平方；最终用有理数大小判断，报告分子／分母及可读小数。零残余分支单独处理；负 h 取 t=0 给 L=0。不用 float 残余、求解器 cost/optimality 或普通四舍五入结果作证书。检查 `0≤L≤U`，否则为实现／输入错误，停止。t r 是可验证对偶见证，解的精确可行性给原始见证。

阶段一仅对保存 v 做一次上述证书。U≤theta 或 L>theta 即停止，不调用 BVLS。跨门才进入阶段二：

`scipy.optimize.lsq_linear(A,-b,bounds=(lower,upper),method='bvls',lsq_solver='exact',tol=1e-12,max_iter=100,verbose=0)`。

不热启动、不列缩放、不改权重／正则化，不添加算法或种子。保存返回的 status、nit、message、cost、optimality、active_mask。有限返回点必须对保存的精确盒逐项检查，不 clip 或容差挽救；越界或非有限则没有新可行证书，保留阶段一界并结束未决。可行时对原binary64返回点计算第二份证书，不要求 success。取 `U_best=min(U_old,U_new)`、`L_best=max(L_old,L_new)` 再作同一比较。求解中间点不另选优、不另作证书。

已读安装源码：`lsq_linear` 先调用一次 `np.linalg.lstsq`；BVLS 初始化与主循环内另调模块绑定的 `lstsq`，且 max_iter 在源码中加上初始化次数。因此主循环100并不等于总分解100。执行包装器须同时守卫这两个入口，共享不可回退计数：**至多1051次密集 lstsq（含首个无约束解）**，每次至多5280行、525列和一个右端；第1052次在入场前拒绝。1051=1+2×525是本轮独立设定的有限分解预算，不是保证完成的算法上界；源码的保守上界为1+n+100(n+1)，本协议不授予该更大额度，也不转入旧101000次LSMR迭代余额。逐次记入口、矩阵形状、开始／结束、异常、rank及返回奇异值摘要，私有原向量不打印。NumPy lstsq 内部直接分解属于本次求解成本，允许其内部 SVD；单独 SVD、秩筛选、截列、额外求解均为0。LSMR调用为0。

绑定本地源：`scipy/optimize/_lsq/lsq_linear.py` SHA `9a6e3a1e3f965510ec49c193d86543f897e6d7a89f2962aec4eb9e0b73af3b9d`；`scipy/optimize/_lsq/bvls.py` SHA `87d6d88776acb4846f32d306821b441a91b97c3b21ef848c45b9a01771cdf183`。实施准备补齐所用 NumPy lstsq 源及二进制运行库身份，独立审查通过后冻结。保持第三方和生产源码原字节，入口计数只在本轮进程中包装；不得复制改写 BVLS 算法。上限异常、分解失败或进程中断均无重启额度；只使用此前已取得的合法返回候选。没有新返回点时保留旧v和旧证书，不从内部帧提取或重新构造候选，按未决结束。

## 3. 三类停止与解释

1. **U_best≤theta：固定线性聚合目标可达。** 若旧点不达而新点达，旧求解对该目标不足。报告可行点与界，不称已得到完整非线性补偿见证。
2. **L_best>theta：固定盒最优线性残余高于目标。** 停止对此缓存盒继续求解；不把该下界外推到完整 ODE、全补偿邻域、其他方向或整个模型族。
3. **L_best≤theta<U_best：未决。** 包括求解到限、没有新可行点或有限工作完毕。L等于theta不能证明严格超过；U等于theta满足可达。按预算闭合，不追加迭代。身份／证书实现异常单列 invalid，不冒充科学阴性。

两档已有非线性核验不提供全盒余项界。本门只对应较大alpha的线性聚合绝对门，不覆盖逐包门、相对消减门、数值余量或未来表现。无论哪类结果，本轮都不自动进入 RHS、非线性拟合或全量 L1；后续动作由root根据证据另定。

## 4. 有限工作、资格与审查

| 工作 | 硬上限 |
|---|---:|
| 科学生产证书 | 2份；每份一次 A z 和一次 Aᵀr 精确矩阵向量积 |
| 生产证书的矩阵标量乘积 | 11,088,000次（含零项的计数上界），另有每份5280项平方、5280项bᵀr、1050项盒端点乘积 |
| 独立证书复核 | 至多逐字节重算上述2份一次，另记同样上限；不求解 |
| 科学 BVLS／主循环／密集 lstsq | 1次／100次／1051次 |
| LSMR、独立SVD、模型／ODE／RHS／H、L1/L2/L3、未来或新FOM | 0 |

Fraction 或等价公共二进制指数的整数累加均可，必须算出精确和；不采用未审查的浮点误差估计替代。用r、s、q、h直接求t及L即可，不必构建全部非二进有理y分量。每份证书的见证z、r、t及精确标量留私有产物并hash，公共输出给输入hash、U/L/theta分子分母、分类、差距、可行性及成本。可不另存完整q，只保存其hash及确定性重算路径。独立审查可从固定输入与见证重新计算，不需要任何模型或求解调用。

纯代数技术资格最多2轮（首查及一次明确技术修正后复查），每轮至多12个手写用例、每例至多4×3矩阵；其中至多1次真实小型 BVLS，max_iter=4、双入口 lstsq共最多16次。技术总上限24例、2次小BVLS、32次小分解；其余全部mock。不使用真实缓存、模型fixture或科学求解来资格。固定覆盖：零残余、正下界、跨门、theta等号；正负q端点选择、非负射线缩放；极小binary64与消去的精确恢复；越界／NaN／hash错拒绝；双入口初始化计数、上限前拒绝、异常记账和无重启；公开输出不泄露向量。第二轮仍失败即不释放。科学执行后修复不得自动重算。

账本分别记录证书、BVLS、全部密集分解、资格及独立复核的真实工作。记录函数内部wall/CPU、证书序列化wall，以及由外层启动前到进程退出的完整命令wall；未测得记null，终态等待时长不能替代命令耗时。有限工作量决定停止，不以耗时预算或普通optimizer success作否证。

root设计采用→独立设计审查→最小脚本与纯代数资格→输入／源码冻结→明确release→唯一执行→独立证书审查→代码／模型／科学三决定。此阶段只写设计和协议，未读私有数组，未运行任何科学计算。default none、已接受模型、val/test关闭及TASK4暂停均保持原状态。
