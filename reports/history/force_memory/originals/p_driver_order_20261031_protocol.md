# p驱动、动态阶数与合法初始化的有限辨别

Run p_driver_order_20261031，discrimination，Campaign force_memory_closure_20261030，2026-10-31。
root决定先辨别，不采用50维候选。实际Astra设计：
reports/p_minimal_closure_20261031_architecture_brief.md，SHA8fa22d9a54d69c23cb4a592fbe40a1e4f22396ae38cf6f7926d401e801d156e2；
数学附件reports/p_driver_order_20261031_mathematics.md，SHAa4ee61d22952e1aea4cfe3f2692877bda5f37a1101de779ba16207edfede79e1。
两份全文是本Run规范；以下冻结来源入口、阶段和人工数值口径。

## 为什么现在做，以及没有释放的工作

父p_history_transfer已普通推送并实核远端6958a48500e7ea98ec3df2aa385416529a1aaaa3，35文件原字节一致。
新heartbeat核S/E/Q原生身份均退出；一个旧PID被更晚Codex进程复用，不当科学Job存活。
当前没有本项目未知科学Job的具体证据。父20SVD/20QR与全部余额关闭，不重跑父结果。
十机F_A削弱固定弱式目标欠求解解释，却未区分隐藏驱动、动态阶数与静态非线性。
本轮先检查一个p内部交换状态的条件可实现性和合法信息下的预测等价边界，不直接新增状态训练。

唯一候选与完整功率账按架构/数学附件。b=0必须从完整系统嵌入父模型，
不能给消元二阶式任意额外p'初值；b²S尺度等价限p/ω子系统，完整ω方程GSp不自动等价。
来源结构不相容只反驳条件精确实现，不否定有效近似；近似需要未来另定误差、时域和评价。

全Run上限：真实轨迹/cache数组读取0、参数估计0、optimizer0、自主ROM/FOM0、模型写入0。
原val/test关闭、旧消费test永久退出、新FOM=0、TASK4暂停；部署100 ms因果δ/ω及末点真p₀不变。
不读取历史真p作初始化，不使用未来真值、事件身份或元数据部署，default none及四指针不变。

## S0：一次静态来源入口抽取

精确11文件及bytes/SHA见results/p_driver_order_20261031/source_bindings.json：8固定入口、3小型谱系记录。
预备盘点只核路径/hash和当前配置：configs/private_dataset.json声明ieee39/ieee39_full.xlsx；
三个预列环境均未找到ANDES包/case，仅用1次一级目录元数据枚举，不是来源身份通过。
S0只能读取这11个静态文件，每个≤1 MiB；先全文件hash/size，再按用途抽取，不追指针。
JSON只抽取精确末级key有关python/executable/environment/version/case/fingerprint/attestation的标量身份，
允许路径字符串；不得因祖先key匹配而输出整段对象或事件列表。配置原件为已绑定静态配置，允许完整私有快照。
源码只输出ANDES版本、case解析、导出字段/控制器保留及fingerprint语句的带行号文本，不调用其中函数。
父报告只作为既有解释/来源声明，不从报告重新计算指标。所有未匹配的历史字段保持unknown。
包路径、配置/生成历史同一性不能由当前文件名推断。

S0输入/源码/release必须冻结；一个正常尝试，started未知或completed均不重放。
公开仅计数、hash、身份层级和缺口；路径、原始身份字段和完整抽取留私有。
S0如果给出精确包/元数据/case指针，root另冻结明确文件/用途/hash并独立释放S1，才能读取其内容。
不能沿发现的任意路径自动搜索或绕过raw HDF禁令。S1当前未实现、未释放。
若入口未提供必要指针，则source_binding_unknown；可完成独立数学套件后闭合，不扩大目录寻找答案。

整个静态来源预算仍为8入口+最多8谱系+最多16 ANDES源码+最多1 case=33内容文件；
目录元数据最多3次（已有1次）；case最多6个明确sheet、每sheet200行，先核原case SHA。
任一后续源路径必须来自已冻结来源引用并由root单独采用；依赖超限或历史hash缺失如实停止。
不执行ANDES import/get_case/load/setup/prepare/PFlow/TDS；不打开raw HDF、任何cache或权重。
最多2种实际源控制器关系作条件比较，未知不按器件名字猜，不把源限幅存在说成train中已激活。
本轮不为找到来源安装新环境或下载替代包；若本地来源缺失就明确保留该缺口。

## M：三份推导与一个六例数值套件

三份科学推导由实际Astra数学附件提供，独立Reviewer核对。M只做六固定人工例，
规范状态(δ,ω,p,r,q)，M=S=g=κ=D=K=λ=1，矩阵逐项按附件，不加载父权重或真实数据。
(a,b,c)依次为(1,0,1)、(1,0.5,1)、(0,0.5,1)、(1,0.5,0)、(2,0.5,1)、(1,1e−6,1)。
0参数扫描、0随机种子、0真实初态、0人工长轨迹。六例是科学工作量，不借工程测试扩充。

每例一次SciPy E=exp(−0.01A)、一次MP100独立E_ref，总6+6次矩阵指数。
MP100精确提升同一binary64矩阵/步长，两路径分别形成0..10次幂；MP幂最后舍入float64，
按atol=rtol=1e−12逐元素比较全部矩阵幂/O_h。不得额外直接算11个时点指数或适配精度。
O_h保持原规范行，不重加权：11点δ/ω共22行，末点p一行，23×5。
SVD阈值=max(shape)*eps64*σmax，记录全部奇异值/阈值/数值秩/原坐标条件数，
理论秩b=0为4、其他为5。数值秩不一致记numerical_rank_disagreement，不改阈值或把近零方向当精确反例。

核能量矩阵恒等式、消元行向量恒等式及b=0一致初值/父块嵌套；不另算父块指数。
未来输出核只用N_out=stack(C_out A^k,k=0..4)，C_out为δ/ω/p；
混合导数行只判代数核，不能用其条件数宣传实际预测可靠。
b=0用e_q核验历史与未来均零；b>0依数学附件精确无混叠证明，数值核查不替代证明。
近解耦的坐标条件数、输出预测敏感性和真实100 ms信息充分性分开，不新增噪声或实际数据试验。
所有非有限统计写unknown/null，不能null又通过。任一科学门失败保存结果，不追加算例或换参数。

## 冻结、审查、停止和收尾

S0与M分别root release；实现/输入/协议/数学/审查hash绑定，完成后核产物和真实进程。
sol承担实现执行；Reviewer与Astra/实现作者分离，只读冻结输入，不把独立上下文当新科学证据。
全Run最多一次纯实现修复；先保全旧hash/失败与已做工作、对账作业，root另释放。
科学未知、源身份缺失、约束不相容或病态不算技术修复，不自动换目标或重复数值指数。

来源判断按当前静态可读、包/case/config身份、历史生成身份、实际工作区间四层分别记录。
L4未读真实轨迹恒unknown；若缺历史身份，只能作精确静态配置下的条件比较。
数学/数值有效性分别判断；本轮无优化，当前健康/充分性not_applicable，来源训练口径仍unknown。
没有真实效应/新数据确认，不以六例PASS当机制识别或架构胜出。
代码no_change或retain_branch、模型retain_evidence、科学continue或unresolved由root收尾决定。
架构当前只采用为可检验假说，不采用为工作模型；任何真实轨迹辨别、额外10次估计或训练均未释放。
完成独立判读、current_state/Campaign/harness与三决定、实际嵌套成本、选择性普通推送和远端核验后进入下一Run。
