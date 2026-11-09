# 记忆转运loop：独立有界静态审查

2026-10-29。实际模型 `gpt-6.1-sol`，独立审查者，未实现这四个文件。仅静态读取新study/controller及两份guard测试定义；不读私有参数、事件、进程或作业，不运行编译、测试、源谱、fixture、CUDA或optimizer，不重审已通过的旧模型数学。root保留科学决定与实际释放权。

**静态PASS：在本次指定决定性路径中，未发现需要先改代码才能冻结的确定性执行阻断。** 这是接线判断，不是新guard测试或真实数值/恢复QA通过；release仍UNKNOWN，真实loop不得凭本报告自行启动。

## 已核路径

- **有限账本与新日程。** study:7–8、64–109定义1848及formal1536/qa108/repair108/replay96；正常journal禁自动repair/replay，actual attempt在step前登记、completed在真实step后，pending按原run隔离保留。正式身份仅A/S/P与30311/30312，每run≤256。新horizon_for独立支持整数1..256奇5/偶8，make_schedule不调用只到192的旧horizon helper；同schedule/随机状态用于三组。
- **128→256与共同端点。** study:182–201最近64次门为clip>16、相对变化>.25至少16次；新增标量用投影后差和max(1,旧值)，与注册一致。64→128真实区间收益、粗细扰动及全部保护才允许一次256。:404–413只在128做正式完整fit和延长决定；:436–445仅三组都COMPLETED且256时追加共同fit256，个别延长只保留面板/检查点，不新增不配对完整fit。:446–449第二种子严格读取128 profile和128优化状态，256不能救回它。
- **研究门和效率范围。** study:217–252对A/P逐一检查S的少漏或风险分支，同时保留A/P/R_N、A96的8s保护、误报0、稳定/p/ω与时机同一T；R_N真实风险上界在OR分支之外。缺有效三组/空T/未决不触发机制保留或第二种子。P_independent_capacity_matched=False，不把冗余路径宣称独立容量匹配。绝对目标同样将R_N风险设为共同必要项。
- **一次平面、无新弱解。** study:44–61只核并读取已经产生的原始plane/public文件固定哈希、status/cutoff/eigh_count=1及源model身份，构造新模型；没有prepare_plane/eigh、NNLS或旧数学suite调用。execution:22–26、45–51将固定model/test/protocol和实际CPU34/CUDA3证据字节入binding，:124–126每worker复核；:94–108另绑定源三资产、plane/raw-public、旧fine缓存和R_N/A96 profile。没有重评旧来源，R_N只读既有profile/时机。
- **六次真实性能资格。** study:271–297每组一次uncached→cached完整12×8 J/反向，记录独占attempt、实际J/全部raw梯度、同步秒数/峰值CUDA内存，0optimizer；按原atol1e−10/rtol1e−8比较。execution:189–215核固定12 ID、范围/精度、顺序、两份实际结果与attempt、完整47/48梯度的数值向量及原容差，不只信equivalent布尔量。:234–236核实际attempt总计≤6且与声明一致；必要组性能失败不训练该组，不补测择优。
- **源兼容与物理资格。** study:300–337先真实CUDA配对和性能，再各组唯一零更新fit297；A0/A96复用两份fine缓存、297身份/历史和原_cross_event，无源重算。execution:137–162显式核path/SHA、身份/别名/原evidence、对齐、双向8s/actual15、全部细子步finite及原scientific/actual门，绑定后续phase.inputs。每组自身原完整fit资格仍保留，缺组不伪造matched合格。
- **真实QA与新loader。** study:137–160使用MemoryTransportROM.from_export而非旧SGM loader，核variant/主history/布局、真实Adam计数和CPU/CUDA/Python/NumPy RNG；u/v及s/d按新format保留。:340–379实现每组18连续与16保存后2次，核非零Adam矩、完整快照、延续参数/Adam和RNG。execution:165–186核三组齐备时真实108/108、六条实际1..18 attempt/completed、唯一variant、QA投影path/SHA及精确内容；允许之后追加formal。失败组的实际调用/隔离状态保留，健康组可继续，不能把缺跑记物理失败。
- **保存、隔离与数据时序。** 正式每16保存，新identity含variant/seed/source/code/schedule。未决模型与真实pending入账，execution:257–265仅允许有对应NUMERICAL_UNRESOLVED、非物理失败、quarantine证据的失败run保持pending，不自动恢复或使其重获资格。study:450在全部全局fit决定闭合后才:462读取train88并报告385/noise88；没有回流选源、延长或第二种子。
- **原生owner与旧终态。** execution:54–76核固定weak complete/summary/source、锁缺失、六个原干净native回执，逐phase完整FILETIME身份只接受GONE/REUSED_PID，live/unknown及查询异常拒绝；保留旧身份而不操作复用PID。父0optimizer/QA未执行以固定闭合证据为依据，不把旧weak未准入改判。:298的uncapped supervise实参与既有签名一致，回执沿用实际launcher/worker exit、Job清理及source前后核对，未新增进程/时间/CPU额度。
- **释放与不可重放。** freeze、worker、launch、run及study各phase入口均要求固定root release；UNKNOWN先硬拒。每phase独占ownership/native/result路径，既有输出不自动重跑；旧976及其他关闭余额不转。正常loop没有额外数学suite、源谱或性能重试分支。

## 效率输出的必要使用边界

study:241–252在每个candidate/A96/R_N共同T内算原E/u，:454–459只用配对已有完整fit成本节点并明确same_cost_winner_claimed=False。输出保留每个节点的T，但没有另设跨模型/跨节点的T集合一致性自动证书。因此root只能在T覆盖实际可比时使用first_attained作同精度成本比较；不可以仅见每个节点target.passed就宣布速度比。若T不同比或无共同能力点，维持当前“不宣称同成本胜出”的范围，只报告能力—成本事实。这是注册的必要解释，不是允许事后改T或增加预测。

## 测试定义与后续条件

两份测试明确使用合成profile/账本、Mock模型/原生回执和临时目录；定义涵盖256日程、最近64界、共同128不被好256救、48参数/plane新loader、1848、release UNKNOWN、sole plane、六QA轨迹、跨模型两范围、实际性能向量以及旧owner复用。没有将这些合成检查称实际科学optimizer、恢复QA或私有数据资格。本次仅阅读定义，未运行，不能据此填tests PASS。固定CPU34/CUDA3和sourceplane通过事实沿root已归档证据复用，没有自行重做。

root应在纯guard检查通过、精确四文件及本审查归档核远端、旧轮终态/锁/成本闭合及release显式绑定后，才实际执行新loop。真实性能、fit297和恢复QA失败必须按原分类保留，不因本静态PASS放宽。旧分类、val/test关闭、TASK4暂停、新FOM0、默认none不变。

## 精确源身份

| 文件 | 实际SHA256 |
|---|---|
| experiments/memory_transport_study_20261029.py | `1cb35ae27317a3ecf4b852895e6eb8564e5517d9b9cb75e14231dcfde16e252f` |
| experiments/memory_transport_execution_20261029.py | `11292e42ef5456afb61ce3b07c43463d740c55704227a57da89c6a98aed91458` |
| tests/test_memory_transport_study_20261029.py | `756f894e44cad150fe25cba08ae18286c52c8912020dcd8bc37fc0d0fe1fcd2a` |
| tests/test_memory_transport_execution_20261029.py | `d846e518bcf5d180798f7dacbc1e4a29c8650d8729dc23a5ad2cba7acd2c81af` |

除本报告未写其他文件。本次有界独立实现审查结束，root保留科学与释放权。

## 首轮guard失败后两处fixture的局部复核

2026-10-29，实际 `gpt-6.1-sol`。root报告首轮实际guard为25 PASS/2 FAIL，两个fixture接口问题使原目标断言未到达；失败已归档 `1a50cd222b3d49a90b3175e52fd6c712a7b85b37`。本次仅只读下列fixture修订，没有运行复查，也没有把原失败改判为已通过。

**局部PASS：两处为stub接口修正，原断言及生产门未弱化。** study test:192只把Windows路径匹配统一成 `Path(path).as_posix().endswith('qa/summary.json')`；:211–213仍要求仅主种子三组、128失败不得由好256救回，以及报告使用新loader。fake返回QA摘要的分支现在可正确匹配Windows路径，未改科学比较输入或预期。

execution test:96–107增加独立fake closed_execution parent，read用严格 `records[Path(path)]` 映射：只提供parent binding.files中的原profile哈希和两份空prediction_files旧evidence，未知路径仍KeyError。生产freeze的profile哈希查找/比较路径未被替换，既有合成sha stub范围不变；新增:110断言新binding保留该profile哈希。原plane、CPU/CUDA记录、无新eigh/NNLS/数学suite断言均保留。这验证合成接口传递，不冒充真实源哈希或metadata冻结已通过。

实际新test SHA256：study `0bd70750950d4a13dc0ce87bb95b055dedc1fb6e839e513afc2e555e02157e40`；execution `d31ebe64afdab8a69e3cb40604ce66ea0a2dd1f6f17bd33f4f69a54382e4205f`。两生产源码实核仍为原 `1cb35ae27317a3ecf4b852895e6eb8564e5517d9b9cb75e14231dcfde16e252f` / `11292e42ef5456afb61ce3b07c43463d740c55704227a57da89c6a98aed91458`，没有代码、门或科学参数修订。

旧25/2、初版SHA和原审查保留。root可在既定授权内决定同27项的一次复查，结果使用新artifact，不能覆盖首失败；本局部PASS不等于复查PASS或真实执行释放。未读私有数据、运行测试/编译或新增计算。
