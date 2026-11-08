# 增长受控边语法：独立结果与集成范围审查

结论：**PASS，支持下述精确六文件作为独立结果文档PR集成到main。** 本轮完成了可解释的固定比较，但没有达到登记的结构选择一致性条件；保留完整负结果，不采用模型，不把本批研究模块或私有launcher随文档合入。root仍须核实际PR diff与预期head。

本审查由未实现本轮的代理完成，读取实际selection、evaluation、native/run记录、completion_summary、completion_facts、qa_summary、原协议与独立增长审查。未运行测试、GPU、训练、数据生成、预测、评价、torch.load或Git；只写本报告。root负责最终科学决定与GitHub实际范围核验。

## 完成事实与科学结论

正式来源为`c9ec666fc7e8dd5ad04f63d9bea3ada33d4ec7a2`。实际selection为train_only、三seed均FROZEN，记录9216更新、72检查点、54排名；evaluation前后冻结验证为PASS，三份选中模型的全部train/合成val时间窗均通过原16事件数值资格。所有54份排名亦为有效；本轮没有需要以中间32/128无效规则解释的观察结果。

固定teacher为候选索引2，表达式原AST为`c*qi`，系数0.15；索引均从0起。实际结果如下，MSE按冻结train尺度归一化，非RMSE。每行train与合成val各16事件、弱强各8，主要窗口为一次100ms初始化后未来`(2,5]`秒。

| seed | 512选中候选 | teacher最终名次 | teacher签名匹配 | train主要窗口MSE | 新合成val主要窗口MSE |
|---|---:|---:|---|---:|---:|
|6201|3|3|否|0.3807425003|0.5864341541|
|6202|0|6|否|0.1222193942|0.2006490250|
|6203|3|5|否|0.2923600710|0.4852747217|

三次最终边界均不歧义，teacher的最终区间组均只有它自身。记录的离散化误差区间很窄，不能把未选teacher归因于该数值误差区间内的排序未决。`subgrammar_consistent=false`与原登记判据一致。恰当结论是：**这套固定子语法、teacher、短时训练目标和512更新选择程序，在三组登记合成数据上未恢复teacher机制。** 不能称结构恢复成立，也不能由此断言模型族不能表达teacher、隐藏机制不可辨识、或真实器件机理被证伪。

选中模型的合成val弱/强/最差事件主要窗口MSE分别为：6201的0.037003/1.135865/2.508184，6202的0.015900/0.385398/1.215590，6203的0.050793/0.919756/1.663089。完整逐事件、1秒/5秒及复杂度记录在completion_facts与原evaluation中保留。数值有效意味着允许解释这些误差，不等于预测精度或模型采用条件已经满足；本轮没有登记的改善基线对照，不能将这些绝对误差写成性能提升。

## 优化与比较范围

全部候选获得相同512更新并使用真实32/128接续，排除了本轮内部“teacher被提前淘汰而缺少后续更新机会”的直接解释；它不证明各候选已经同等收敛。completion_facts从既有checkpoint提取的18条实际0.5秒小批次训练loss中，最后32步均值均低于前32步，比例范围0.859347至0.969091。该事实提取由既有记录的CPU读取生成，本审查没有重新反序列化或独立重算训练loss；这些趋势支持保留优化充分性的疑问，不证明延长训练必能恢复结构。

与上述拟合loss不同，teacher候选的train`(2,5]`排名MSE在128→512分别从0.214215→0.519230、0.118218→0.176154、0.239595→0.416605。两种目标一个用于0.5秒拟合、一个用于长时排序，不能把后者上升写成拟合loss发散，也不能把前者下降写成长时预测已经改善。没有额外优化、状态可辨识或目标对照，现有结果不能分离短时训练目标、可训练初始化、隐藏q、优化路径与结构本身各自的作用。

三seed同时改变合成事件和优化随机流；它们是三次登记的完整程序试验，不是固定同一数据后的纯优化重复。数值误差区间不是统计置信区间，三次结果不提供总体成功率显著性结论。相较旧轮同时改变了受控语法、teacher/数据流和不早淘汰程序，不能把新旧差异单独归因于增长筛选或完整512比较。旧teacher数值失败与旧6102误选保留原判定。

此前增长审查证明的是固定有限参数下向量场至多线性增长，适用范围及端口限制不变；本轮数值通过与该解析性质相容，但不构成全时有界或L2稳定证书。结构未恢复也不推翻该增长论证。

## 来源、隔离与证据范围

实际native_run为COMPLETED，native/worker/launcher均typed整数0，Job0、worker GONE、cleanup_verified=true、orphans_cleaned=false；summary记录锁清、162项来源前后及当前字节相同、144检查点文件SHA已由root核验。正式wall为32077.1934998秒，Job user/kernel CPU分别24505.265625/5598.296875秒。这些计时与QA或完成后只读事实提取分别记账，不能相加后冒充一个原生正式Job计时。

本审查对completion_summary引用的五份原始文件逐项核了SHA及字节数，并核qa_summary的四份主要引用SHA，均一致；未重复核全部162来源或144检查点，也未把root的核验称为本审查独立执行。实际QA记录为14通过、0失败、36次CUDA资格更新与2次CPU资格更新，正式更新为0；其PASS只覆盖该来源的相关QA。

selection为train_only；固定stream0生成train、stream1在18最终端点核验后生成一次新合成val，只评价三份train选中模型，未比较全部18候选的val。本轮结果及既有静态入口证据一致地记录项目原val/test未访问；本审查没有独立进行操作系统级访问追踪。无val选模型或继续训练证据，不能把一次新合成val改称原项目val，也不能借本轮开放原val/test。TASK4仍暂停。

completion_facts含18模型的既有计数/Adam/trace摘要与实际loss趋势、54排名、三选中模型各时间窗及225项本地路径/hash引用；`relative_to=run_directory`与绝对run_directory说明解析基准。它保留本地训练状态与数组的复现路径，不等于这些大文件已上传main；读者复现训练/预测仍需归档源码及所引用本地资产。未记录的事实提取PID/CPU明示为null，不能补造或算入正式Job成本。

## 建议的精确main范围

本轮建议只合入以下六文件：

1. `reports/bounded_edge_grammar_20261022_protocol.md`
2. `reports/bounded_edge_grammar_20261022_conclusion.md`
3. `reports/bounded_edge_grammar_20261022_result_review.md`
4. `results/bounded_edge_grammar_20261022/completion_summary.json`
5. `results/bounded_edge_grammar_20261022/completion_facts.json`
6. `results/bounded_edge_grammar_20261022/qa_summary.json`

它们承担登记问题、完整负结果、优化证据、审查与归档导航职责，属于结果文档进展。新`bounded_edge_grammar_20261022.py`、其tests、复用但尚未合main的finite入口及私有launch/make_binding均不在这个PR中；不宣称这六份文档构成可直接运行的自包含工程包。原始来源和资产应在归档分支/本地引用中保留，禁止随文档默认采用模型、恢复旧失败结论或宣称模型族证伪。

## 六文件最终核对

已读取最终conclusion，并将其19项合法池、teacher2/0.15、选择3/0/3、teacher名次3/6/5、主要时间窗/弱强/最差指标、9216/72/36/54计数及优化趋势与实际selection/evaluation、三个摘要逐项核对；未发现影响本六文件文档集成的矛盾。conclusion准确保留结构恢复不成立、模型不采用、数值资格与科学采用分开、旧失败不改判及本地资产复现边界。末句horizon_pair为下一轮前瞻计划，不是本轮执行或成果；本次PASS不为它提供QA、训练结果或新授权。

最终核对时其余五文件SHA256如下，均为本次实际读取的字节；本报告自身随后定稿，最终SHA由root在精确PR范围检查时记录，以免自引用。

| 文件 | SHA256 |
|---|---|
|`reports/bounded_edge_grammar_20261022_protocol.md`|`e2beca01b2eed8ffbd812e559ee4b77d0fa515f0edd56bb0430cfcbedad4c4c8`|
|`reports/bounded_edge_grammar_20261022_conclusion.md`|`425e41961be1aeb1d025ed332414bc17cbb541c4fa72ce055b1db64d87f79cdd`|
|`results/bounded_edge_grammar_20261022/completion_summary.json`|`9be3454defdea1e0171b75b6decd7e0f15434a6e2301653d91edeaa11d2a5951`|
|`results/bounded_edge_grammar_20261022/completion_facts.json`|`70c5bd6f7ea196298796f79db769eb543125321168f2b5513bb962dc486196f9`|
|`results/bounded_edge_grammar_20261022/qa_summary.json`|`19303aa8d8978bd34b4e0a7ec3dab1dd5fccf89f11777e1a522d28ece9180f59`|

这是结果证据与准确主张的集成资格，不是结构恢复、模型晋升或独立可运行工程包的资格。精确范围保持上述六文件，不集成依赖尚未合main的finite入口的研究实现、测试或私有launcher；不更新旧source-bound增长审查。本次未运行任何测试、模型、数据或评价，亦未检查实际Git diff或远端状态。
