# 时间目标研究：独立实施审查

2026-10-20，独立代理`/root/fixed_edge_final_review`。本代理未参与本轮提案或实现。本次只读封版候选runtime、runner、三个测试文件、私有launcher/make_invocation/QA driver、协议及必要直接接口。没有执行pytest、训练、成本测量或新诊断，没有读取私有权重、HDF5、val/test，没有递归委派；仅新增本文。以下是实际代码路径的静态意见，不是实际QA通过。

## 公式与共同计算

runtime将每条长轨迹固定在endpoint10，使用TensorEventStore提供的真实n；尾长度以(n+4)//5计算，按全局未来索引跨ragged segment累积，返回0.5时间均值+0.5末段均值。每条轨迹只在起点编码，状态跨segment保留；没有中途真值重置或补尾。新加的规范ix检查要求每segment恰为按实际n升序留下的活动行，并核累计coverage等于n，能拒绝总数相同却破坏时间顺序或重复行的伪segment。

O/F及T非第三相位取原时间均值，T第三相位才取加权值；风险、物理尺度、辅助项和系数保持原定义。local使用独立原samples的0.2秒batch，hidden一致性使用原相邻history，torque使用原samples位置，没有复用长轨迹末态。one_update检查完整目标、梯度、clip及Adam后状态有限，异常finally清cache。静态未发现公式或辅助项计算改变。

synthetic测试定义覆盖独立手算权重、n=1、20%整数边界、不同n及跨segment尾段的状态/参数梯度；O目标值、旧组件、全部梯度和真实Adam更新与原路径对照；T与独立单事件积分对照、辅助项各自梯度和三种历史编码分离；非有限/异常不执行Adam、cache关闭及非规范segment拒绝。CPU/CUDA参数化不等于实际已运行，结果仍待统一原生QA。

## 生产入口、来源与恢复

runner第40–45行把三源model/training/checkpoint的实际SHA锁到对应B512，并检查seed/variant/source_global_step；新模型只从该源加载，再严格启用delete7缓存，不调用改变表达式的build_variant。第121–137行统一创建空Adam、赋相同RATES，然后恢复源CPU/CUDA/NumPy随机状态和coverage。prepare核源data身份与完整385一致，日程仅调用一次next_samples；逐行保存O/F/T时域映射及各组真实long_counts、共同local_counts和前后RNG。训练第188–197行校验采样与实际区间，T仅step%3==0启用尾权。没有768、换seed或自动科学重试入口。

正式恢复继续复用原子identity checkpoint链，恢复后原地启用delete7缓存，不重建Parameter或清Adam。只允许root明确外部中断授权、对应seed/variant、32整倍数安全步及既有progress差0–32；唯一恢复/1200和各模型2000或4000由launcher累计门禁承担。root仍须用真实PID/创建时间和原生失败回执证明外部原因，不能只凭布尔授权JSON。新端点身份保持source_parent_stage_step=512与新stage_step分离，不能称连续Adam续训。

原生监督沿用已验证PS/native_job，要求worker创建身份、两层退出/完整回执、清理GONE及源前后相同；独占lock与完整native prior集合防重复队列，清理未决拒新launch。make_invocation明确收集train叶文件、三源、直接代码依赖、命令/prior和实际admission/schedule/recovery/model身份，不对共享HDF5作整体hash；源与命令冻结及完整绑定仍由root执行。三个QA测试文件中native测试依赖私有launcher，不能把这三文件整体称公开clone自足CI。

## 当前必须在执行前闭合的事项

1. **实现过程费用须进入实际门禁。** 当前读到的launcher.prior_cost_gate只累计native回执，再加3800支持，未消费BASE/implementation_cost.json。root已发现worker静态/读取实耗并正在核账；该项不能只在结项总账列出。须固定实际费用及身份、纳入每次命令binding，并同时扣工程4200和阶段42000，在启动前拒绝遗漏或不合法记录；避免与3800hold或native子进程重复收费。此项闭合前不授予真实QA启动准入。

2. **实际B源初始等价尚无生产检查。** new_stage加载、启cache和恢复RNG之后，measure_cost第295–310行即调用首更新；当前未比较实际source的generic场、encoder、全部state_dict及参数requires_grad与新stage。synthetic旧目标/模型测试不等于三份实际源已核验。建议在丢弃cost试算首更新前，以固定QA历史/状态对source的SymbolicClosureROM generic与新stage缓存作等价检查，保存与恢复CPU/CUDA随机状态，不增加优化更新、不改源、不看科学预测分数。实际覆盖哪几个seed/变体须明确，不能以仅6101成本试算宣称全部三源已核验。该检查作为成本/准备门槛，QA定义本身仍不代表源实际通过。

3. **恢复witness的实际时域需root明确。** measure_cost第298–320行的cases顺序按horizon再panel，当前保存后“下一更新一致”固定cases[1]，即1秒short_unstable。F/T完整15秒反向/Adam会另测两次，synthetic恢复测试虽传15秒但真实夹具未来只有十几步；因此当前不能称恢复后的下一次实际15秒更新已验证。若兑现方案审查中的“下一实际长更新匹配”，可在同一12次内把该witness改为F/T15秒long（O5秒long），复用该组合两次并保持其余每组合两次，不增加预算；须root在执行前确认并冻结这个选择。

## 评价及费用准入的已核路径

cost确实按三组各时域实际最慢值、每32步保存及加载估价，乘1.25，九组≤28800、O≤2000、F/T≤4000，评价九组≤3600且每队列≤400；flat admission.json由生产train直接消费，不接受嵌套cost报告代替。成本试算设计12次包含恢复分支的匹配更新，不把丢弃试算转成科学源；后续完整档不容纳则不启动九组，不自动删T。source/data加载、save/restore、实际15秒反向、显存及首次缓存成本是否完整，仍须本轮实际回执验证，不能从代码定义宣布准入。

正式评价每端点一次fine/coarse预测，原1/2/5/8秒、真实15秒资格、分类与物理误差保持原接口；新tail指标直接消费同一fine数组，不增加rollout/encode。固定cohort身份以哈希锁定原366/新19，不依前缀。temporal_metrics第350–358行记录finite，尚未自行判数值资格，因此正式提取必须把尾段指标与该事件真实15秒资格同时判定，不能只筛finite后宣称全范围合格；原始失败/未决保留。新尾段和≤5秒/>5秒是解释性分项，不替代正式候选门槛。

## 当前意见

除上述执行前事项外，未发现另一个确定数学或生产接口错误。当前不称QA通过；成本门禁未闭合前不授予实际QA，真实源等价和长更新恢复覆盖须在成本阶段完成前成为明确条件。后续若这几项由root落实，复核相应最小diff即可，不另开规划或审计链。工程PR宜限制为通用runtime、对应纯synthetic测试及必要说明；不整体合并runner/helper，不带新默认模型或科学候选。

本次支持费用采用600秒保守份额，嵌入既有3800支持hold，不新增预算、不在hold外重复收费。科学决定、实际费用核对与启动授权由root负责。

## 同轮三项修订的有限复核

root在任何实际QA前采纳以上三项，并在协议末尾锁定费用、最长更新恢复、真实来源等价及尾段资格口径。本代理只读核对最终最小diff，没有运行程序或读取实际模型资产。runner当前SHA为`21156f11fcde2c86d7bf65aff8503829d6f0e52485efa8c9d48e4a3ac3f1a1d3`，runtime为`dec2e8063c9dc662de6c8aca8c77a6f4dfcb3a538cfb1923ec9de7ab1a6bc79e`。

1. launcher.command_gate现在将固定`BASE/implementation_cost.json`列为必须绑定文件；make_invocation也实际纳入该文件。prior_cost_gate第85–91行读取合法qa_seconds加入工程累计，并在第115行随其他费用同时检查4200及42000。当前记录47.1277233秒，公开副本与私有文件SHA相同；root/审查只计3800hold，未重复纳入该数。普通生产入口先验证完整binding，因此缺文件不能以辅助函数默认0绕过。该费用门禁阻断关闭。

2. runner.source_equivalence第291–326行在cost首更新前实际循环三seed×O/F/T九个新stage，与各seed的SymbolicClosureROM generic来源比较完整state_dict、requires_grad、encoder和field，并要求空Adam。使用独立固定Generator创建零更新QA面板，不消耗正式采样流；finally在成功/异常时均恢复外层CPU/CUDA RNG。measure_cost第342–343行实际调用并保存结果，新测试定义覆盖九次调用及成功/失败RNG恢复。该覆盖路径静态闭合；“三来源已实际通过”仍须等成本阶段真实回执，不能由代码存在推定。

3. cost_cases第328–332行把第二组合改为O5秒long或F/T15秒long。measure_cost仍以第一组合首次更新后的非空Adam检查点作来源，第二组合两次分别为reference和restore后的同一更新；剩余组合补齐，每组严格12次。新测试逐项核六个唯一组合各两次及实际总数12，未增试算更新。该最长实际更新恢复条件静态闭合，真正CUDA反向、Adam及RNG匹配仍由统一成本执行验证。

root同时在协议中明确尾段解释指标必须join同事件真实15秒数值资格，任一失败保留未决，未改变原正式评分。本次无需再改冻结数学目标、科学分支或预测程序。

**本节取代前文暂缓准入状态：上述三项已在生产消费者中静态闭合，本范围通过，未发现剩余确定阻断。支持root冻结最终源码、精确命令、完整直接依赖及费用身份后，执行已登记的一次统一原生QA；实际QA尚未运行。** QA通过也不直接授权正式训练，仍须实际成本、九组完整档及来源等价/最长更新恢复通过后由root决定。唯一明确实现修复限制、科学不重试及真实回执规则保持；工程PR仍仅限可选runtime、对应synthetic测试和短说明，不整体合并runner/helper或默认模型。

本次是原实施审查的有限补充，使用先前600秒支持份额的剩余部分，不重复增加费用。
