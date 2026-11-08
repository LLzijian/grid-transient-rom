# 先核真实历史来源，再检验共享p常数的时间转移

Run p_history_transfer_20261031，discrimination，Campaign force_memory_closure_20261030。
2026-10-31；状态DESIGN_FROZEN，root采用实际Astra方案；先只实现/审查S来源阶段。
规范：[p_history_transfer_20261031_scientist.md](p_history_transfer_20261031_scientist.md)，
SHA256 a51c2292179967ad4ae05c69f2c68500aac51225b57a40cd14ad5d48ff8c560c。
其§1–§8全部作为本Run规范；以下补齐资产、阶段与分支量词。尚无实际来源载荷核验或估计放行。

## 采用理由、历史与不变边界

父p_form因40/80 ms纯未来完整覆盖不足，在估计前停止并远端归档55057a150ce658e987d50f86d5a1bfaaad897ba2。
旧未用10次机会关闭。新预测是：仅由共同历史估计的常数，能否解释从同一部署点开始的未来p响应；
全分析段估计作为区分历史识别不足与整段闭合残差的固定对照，不靠改名重跑旧20 ms M3。
root先核信息和覆盖，再投入后续实现；S不构造X/y、β、残差、rank或滤波。

唯一train fit297=267稳定+30失稳，缓存SHA3058c06fe23479a2c494203bd379ca495c54953b6d8b993a6591c26905cc4252。
原val/test关闭、旧消费test永久退出选择、新FOM=0、TASK4暂停。部署仍100 ms δ/ω及末点真p₀后共享自治ODE。
真历史p/未来真ω/时间元数据仅离线辅助，不进入部署；不改40维状态、坐标、模型、四指针或default none。
旧β与全部train已暴露、旧M3unknown保留；本Run是探索性前瞻诊断，不是独立数据确认。
上游tm标注及p物理单位限制不变，数值坐标内的解释不认证SI功率/能量。

## S：一次有界信息资格

精确允许输入见results/p_history_transfer_20261031/source_bindings.json：仅父prepare/manifest及其绑定来源，
fit缓存生产prepare与其24项代码哈希、旧M0/M4已有证据、明确原来源的公共metadata和生成/适配代码。
root只读规划核查已确认生产prepare绑定该fit缓存、297 fit ID及24项生产代码均与当前文件哈希一致；
此规划核查未打开缓存数组。S仍须重新核精确资产、来源家族和原回执，不把规划文字当真实结果。
生产prepare SHA a6f906a5e933f1a39af76546826851bd92e534b594e1aeaae3d027f85c64199a。

来源链须绑定GENROU.tm→导出Pm→Pm_raw减冻结base.pm_star→逐时刻x→原样float64缓存。
导出值是已有采样/插值的记录真值，不能称本轮重新核验了连续高阶解；历史p不得来自反推或填充。
现有代码与生产prepare真实SHA逐项对齐；不能只凭当前代码、非零波形或“M4曾运行”判真值。
cache只读allow_pickle=False，先全文件SHA，再取已绑定297成员前13点ω/p、整体shape/dtype/长度和metadata。
原分母、节点/坐标/基准、历史11点及未来[10,12]均核验；不生成X/y或任何估计/响应结果。
没有独立历史快照时记not_available，不为此加载整批预测；完整缓存来源链仍须独立成立。

允许对原索引文件只提取已冻结fit ID的metadata字段；不按其他划分标签查找或读取轨迹。
不调用read_train/load_events等重新枚举数据的加载器，不打开原始HDF events，不作通用目录递归或补数据。
父p_form manifest已经保存且核对共同β0/p_scale，可通过原回执及参数哈希复用，不为复用常数绕过旧入口。
若确需载入模型，仅用已核专用overlay loader及原资格，不能改旧源码/门。

S按Scientist§2核0.01 s与切除后恢复域，并保留实际clear/delay/record_start/t0偏移的证据层级。
缺字段被or 0.0回退不等于实测0。能证切除后但准确delay未知时，offset记unknown，范围称“既有记录首100 ms”。
若连切除后域或真历史p来源都不能证明，依赖项停止；不移动x[10]、不重采样、不转读raw events。
只可用新的显式来源修订补充已授权的明确证明清单，不能默默扩张数据权限。
S来源、时间、全297覆盖及独立审查均通过，root才释放后续实现和E/Q计算。

## E/Q：预先冻结，S通过前不实现完整runner

u_j=(j−10)*0.01 s，部署点固定x[10]。βH只用x[0:11]内两个40 ms窗[0,4]、[4,8]估计；
末20 ms不入估计，仍进入完整100 ms历史响应和完整历史20 ms附表。
βA分析段为历史加稳定未来8 s/失稳各自真实未来（最多15 s），即x[0:11+n]；不使用稳定8 s以后的标签。
H/A两目标均锚定原索引0，不扫相位；40 ms [8,12]为跨部署点窗，单列cross_t0，不能称纯未来。
各目标每事件等权、事件内窗等权；A的history/cross_t0/future权重指各类窗行权重之和，跨界不拆成纯未来。
全297未来首20 ms评价独立固定[10,12]；未来持续响应覆盖各自完整真实范围，绝不以历史80 ms冒充未来覆盖。

每机每目标一次主SVD，最多20次新估计；同目标独立QR最多20次，仅求解器核查，不成第二组候选。
逐项立即封存β、输入哈希与回执；Q只读冻结β0/βH/βA，不重新估计。
原atol=rtol=1e−12、SVD/QR/法方程/重构、rank/κ/η、符号和解析正负/零a门按Scientist§4–§5。
沿用已审派生统计有限性规则：RMS/差/nu/epsilon/阈值或解析核查误差非有限则原分母unknown，不能null又判通过。
无迭代optimizer、新自主ROM rollout、模型写入、新FOM；不重算旧M2/分类/时机或增加参数目标。

分支逐机，不以机器投票或一组覆盖补另一组。HistoryGood明确要求完整历史20 ms弱式也compatible，
mixed/unknown不使该充分条件为true；共同80 ms三L弱式/局部和完整100 ms持续响应均依设计全部相容。
C_H的β0可分辨前提冻结为：至少同一主组历史共同80 ms三L弱式均resolved，或至少同一主组未来首20 ms弱式resolved。
FutureGood、FutureFail、AllTargetGood及F_A按设计§6；F_A历史定位的弱式/局部要求该组三L全部满足。
H不可辨识/A可辨识是独立信息标签，可与F_A并存，不隐去无效H；正向采用C/转移失败须各自识别和符号门通过。
F_A可保留非法符号的宽无约束最优参考，但不得推为所有合法常数对未来轨迹目标无解。
H/A目标范围不同，不按目标残差大小直接排名；采样表示差不是连续真值误差上界。

## 执行、停止与归档

先S一次正常资格，再E固定20槽、Q一套测量；每阶段独立root release，不自动重试。
纯实现错误全Run至多一次修复，须先保存旧新hash/失败及已完成项，root单独放行；已估计结果复用，20上限不重置。
来源unknown、不可辨识、符号或科学门失败不算技术修复，不触发扩窗、重选系数、删样本或训练。
未知Job先对账。S执行包装/原生身份放private_data/p_history_transfer_20261031_execution，不能预占工具结果目录。
私有数据/β/逐事件身份留private_data/p_history_transfer_20261031，公开聚合/hash写results同名目录。
CPU float64单线程；读/估计/滤波/原生wall按包含关系记录，未知CPU/LLM/项目总成本null。
五轴、三个决定和确认范围按规范；任何β不写父模型，不晋升。来源合格本身不算机制发现。
完成独立判读、更新Campaign/current_state/harness、选择性提交普通推送及远端核验后，才进入下一Run。
