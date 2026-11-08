# analytic_p_backward_qualification_20261031：独立审查

**冻结设计审查通过，可进入实现冻结；实际执行仍待冻结实现审查。** 两个显式overlay候选与其原版配对验证数值行为保持，数学/因果/序列化及自主预测资格与原科学准入分开。设计没有用父probe的小初值差替代完整资格，也没有解锁旧M3或授权模型晋升。

日期：2026-10-31（Asia/Shanghai）。task_id：`/root/review_checkpoint_audit`；role：Reviewer；configured_model：`gpt-6.1-sol`（root任务合同）；server_resolved_model：`unknown`（工具未提供）。本次按heartbeat任务合同仅向root汇报；只写本新报告，旧诊断及精度probe报告冻结不动。报告按项目anti-defensive-writing原则陈述结论、范围和依据。

本Reviewer未读取私有模型、事件、预测或真实计算载荷，未运行训练、拟合、ROM评价或资格套件；实现仍写入时没有读取。当前输出帮助root判断是否继续实现准备，不提前宣称任一候选已合格。

## 1. 实际冻结输入

| 设计输入 | 实际SHA256 |
|---|---|
| reports/analytic_p_backward_qualification_20261031_scientist.md | 7E174327E77A8364FDE0DAB4892CD15A094C375169F7D963918E6CA466A9E496 |
| reports/analytic_p_backward_qualification_20261031_protocol.md | 120FEEC1543150B699C363ADC90EB444E158BDFD373DF62D61F481B31A1205DE |

两者均与root交接值一致。实读HEAD为 `BE39CFB5F84C079753D3BD250EFC94E8A6177ED5`，与root报告父probe远端归档提交一致；本Reviewer未重新读取私有archival_verified或重复查远端。真实输入、父proof、模型、源码闭包及基线缓存身份仍须由正式实现和执行manifest核验。

## 2. 设计逻辑核查

**候选身份与加载。** 唯一A96_analytic/A16_analytic采用analytic_p_backward_overlay_v1，父模型加overlay构成复合身份。先原严格loader完整通过，再独立核数学来源、deepcopy并只替换p_backward；其余state_dict、flags和原对象不变。A16不把改buffer对象伪装成旧MemoryTransport格式export，从而保留原fixed-buffer/config/plane断言，而非放宽旧loader。保存/重载走显式overlay，评价deepcopy与缓存必须保留补丁及复合身份，避免回落父版预测。

**数学和因果入门。** 固定binary64 Q、80/120位解析与独立expm参照及原1e−12门，零a精确极限、原历史两步长和297事件反推均明确定义。原严格加载、参考构造和实际历史门失败均停止依赖项，不放宽门。初始化新buffer实际使用、末点硬钉、非末点历史p占位不影响、空/热缓存一致及原对象不变均有具体核查要求；不改变合法100ms信息或ODE。

**完整paired。** 正常两份新完整fit297评价，每候选一次，CPU单线程float64、future0.01/0.005s、history0.0025/0.00125s及原sgm评分/分类函数不变。稳定1/2/5/8s与失稳真实区间分别评分，8s分类完整预测与真实区间分类并列，未决留原267/30与297分母，不用短真实记录替代完整分类路径核查。旧A96/A16基线复用已绑定完整缓存，基线不足不自动重跑。

**跨实现误差门。** 新旧fine对fine、coarse对coarse分别核原COI/共同转速差与额外p归一化RMS差。每个主指标独立定义u_score=|fine−coarse|，risk、共同omega RMSE和归一化p RMSE均由同队列/时域/真值/函数计算，不能把risk扰动用于RMSE。配对主指标门是当前数值分辨范围内一致性，设计没有把它称统计等效证明。

**事件身份与T。** 两时域合格检出、漏判、误报、数值未决、未覆盖和判据未判定集合分别冻结并逐集合一致，不能用计数抵消错误替换。T_base按各自原版完整可定位集合冻结，要求全覆盖、truth相同、逐参照E门和逐事件确认时刻门；u_time/v_time以秒定义，不与u_score混用。空/缺事件不通过，不能缩分母。原A96/R_N保护并集、固定T_anchor及全部科学门另表，A16原缺覆盖不会因实现资格通过而被追认科学准入。

## 3. 科学范围与有限工作

本Run只使用已消费train fit297、两个既有权重的数值候选，optimizer=0、共享参数拟合=0、新FOM=0，原val/test关闭、TASK4暂停、default none及所有指针不变。新自主预测是本轮明确登记的两次完整paired资格，不能误沿父probe的零rollout口径；也不增加第三个模型或扫描参数/步长。

协议总预算至多4份新完整paired，含正常2份及全Run一次纯实现修复最多重做受影响2份；root当前只释放正常2份。真实数值/科学门失败不是修复许可；修复前须对账和root单独放行，保留失败、原因及重复成本，不借父Run关闭余额。单候选失败可保留另一独立结果，但通用数值实现采用须两者全部implementation_qualified，不能只取通过者。

成本按构造、overlay、初始化、积分输出、总wall及修复范围记录，旧基线推理不计新增；设备/精度/时域/批量/事件口径不全则不作同精度成本比较。明确日志与独占结果目录分开，未知Job先对账，避免重复启动。

父Run原1e−12门失败与旧M3 unknown保留；p按既定数值坐标解释，tm物理含义与标幺基准的来源限制不会因数值实现合格而消失。旧beta即使引用也是已暴露估计，不能当本轮重新拟合或独立确认。后续p形式归因、常数适配或新状态均需自己的前瞻协议，本资格不自动授权。

## 4. 当前决定与实施核查重点

未发现需要扩大测量、训练或无限确认才能解决的实际设计阻断。冻结实现审查应重点核：复合身份贯穿overlay重载和评价缓存；原loader/评分源码不变；基线身份集/T_base先冻结；fine/coarse各指标的u_score、逐事件分类/T及原科学T_anchor分开；失败不删分母；正常仅2份paired及不可盲目重放。

设计阶段建议代码no_change（尚未审实现）、模型retain_evidence/no_change、科学continue准备。implementation_qualified、science_entry_eligible和晋升是三个不同判断；本轮无论结果如何都不直接改变受保护模型或默认指针。

## 5. 首版冻结实现审查：暂不执行

**设计仍通过；首版实现存在启动前必须关闭的合同问题，当前暂不放行。** 本次追加前本报告SHA256为 `51C30409E919CB11299E01D62E48E843FCD080B39AF2C34235D70105E11FC739`。root已确认暂缓启动，尚无prepare或预测执行；以下为未执行候选代码审查，不是已执行科学Run的技术修复或预算重置。

| 实际读取的首版输入 | SHA256 |
|---|---|
| experiments/analytic_p_backward_qualification_20261031.py | 0C6A3DDD18CA4A628984038DDAD9BE4B7D7275E6483F0F0AA58F193833E2FED3 |
| tests/test_analytic_p_backward_qualification_20261031.py | C02813F83B51048CB4903459C8C0619467A4064C738E496BE2BB3A4B911CF4F1 |

两者与root交接值一致。Modeler报告12项人工测试通过；Reviewer已读取实现、测试及必要原公开helper源码，未运行测试、完整资格、prepare或读取私有载荷。旧报告不变，问题均已具体反馈root。

### 5.1 必须关闭的问题与最小修订

1. **新候选数值未决被误作基线复现异常。** baseline_description原第294–296行无条件要求profile各主值非null；evaluate第469行把该严格基线函数用于新候选。当新paired出现应保留的数值未决/主值null时，会抛baseline reproduction错误、缺完整qualification/complete，并使summary无法完成，未按协议输出原分母上的未通过资格。旧基线严格要求有效正确；新候选须单独允许数值未决，将相关official分值置null并标原数值资格false，保存逐事件未决、分母与完整结果，不把真实数值门失败变成可修实现问题。
2. **父归档只核相等，不核固定父提交。** verify_parent_archive原第303行可接受任意非空local_commit==remote_commit，未要求协议指定 `BE39CFB5F84C079753D3BD250EFC94E8A6177ED5`。最小修订为固定父commit并严格核local/remote均等于它、ordinary_push_verified is True，继续读取真实archival_verified.json及绑定其哈希，不造marker。
3. **直接科学/来源helper未完整纳入源码绑定。** source_closure首版未包括实际调用的memory_transport_study_20261029（原efficiency_target）、checkpoint_selection_decision_20261030（集合/时机）、p_port_diagnostic_20261031（独立反推）及closure_probe_20261030（严格父loader）。应把这些直接语义依赖加入当前闭包及verify_prepared核查，与既有sgm.code_identity覆盖的原评价来源结合，避免关键门或加载路径改变而未被本Run身份发现。
4. **增加了协议未规定的coarse category全等硬门。** compare_candidate原第423–425行把fine/coarse category_changed_ids都要求为空。冻结设计要求完整category/window_settled变化列出，影响已冻结集合才不通过；coarse的settled/unsettled变化即使fine冻结集合未变，也会被此额外硬门否决。应保留完整变化报告，使用冻结ID集合及原数值资格作为规定硬门，不增加coarse category全等门或改科学阈值。

root已将上述最小启动前修订交Modeler，保留旧哈希后重新冻结；待新哈希再复审，当前不执行真实资格。

### 5.2 已核且未见问题的部分

overlay_v1先原严格loader，再验Q/数学回执和原state身份，deepcopy只改p_backward，动态绑定专用overlay export；回转重载不伪装旧格式。独立analytic80构造、120位/高精度expm与SciPy门不是从旧expm输出择优；源代码末尾main guard顺序正确，因果入口的实际trace、早期p占位、热缓存、末点硬钉和原1e−12核查齐全。

prepare单writer先存基线ID/T_base与原科学anchors，evaluate按候选独占目录各正常一次sgm.evaluate，无共享写结果和自动重试；已存在started而无完整回执即拒绝重放。正常最多两paired，额外修复预测不由CLI自动授权。两候选均完整回执后summary才汇总，两者是否实现合格与科学准入分开。

cross_interval保留细子步有限性，fine对fine/coarse对coarse的COI、共同omega与p误差计算符合原尺度；真实区间与8s分类确认端点独立检查。risk由逐事件_point_error原risk函数汇总，omega/p RMSE从各自SSE和分母汇总，五个主分值的u_score各用自身fine/coarse差，不用risk扰动代替RMSE。固定T_base不缩集合，E门与逐事件fine确认时刻门均按秒计算，原科学T_anchor及检出并集另表，未见公式或组合数错误。

细子步NaN的cross_interval人工案例能阻止资格，但不足以覆盖5.1的新候选null全流程；新修订应增加该类有效负结果的人工路径核查，不需要真实预测或扩大科学确认。当前建议代码retain_branch等待修订、模型retain_evidence/no_change、科学continue实施准备；数值候选是否合格仍unknown。

## 6. 修订版复审与正常执行决定

**修订实现审查通过，§5四项问题关闭，无剩余实际启动阻断；可按root当前授权启动正常两份paired。** 这不放行额外修复预测、不改变任一冻结门，也不提前给候选实现资格或科学准入。

本次追加前本报告SHA256为 `A29B7B5AEDBEBB94405AAC81B5E7BF10439D45A960D96AAE8C49AC91DD35877C`。

| 本次实际读取的冻结输入 | SHA256 |
|---|---|
| experiments/analytic_p_backward_qualification_20261031.py | EE8429D0FE558B73550ED9D47A04B47FDA957E676BDDD3A3CCDB48D08F10BC1A |
| tests/test_analytic_p_backward_qualification_20261031.py | E21BCE00D7A529DA0A77301E2E67BB7C96481CE0F6660E26F89D2646889731E4 |
| results/analytic_p_backward_qualification_20261031/prelaunch_review_revisions.json | 233336AB0B25F3428C3C2839F8A8C7C029F9FC593E54B058B11E99AB8C69745C |

源码及测试与root指定值一致。公开修订登记保留首版哈希、四项问题与新哈希，声明无prepare/新完整paired、真实修复额度未消费，科学门/阈值/候选数/原loader不变。其状态REVISED_PENDING_REVIEW与本次源码审查前状态相符；实际放行由root另记当前报告哈希。Modeler报告19项人工测试通过；本Reviewer已读测试，没有重复运行计算或读取私有载荷。

### 6.1 四项闭合证据

- baseline_description保留strict_baseline=True用于原基线复现；evaluate新候选明确用False。数值未决时official相关fine/coarse分数置null，有限路径重算仅放诊断字段，不用它替代official值。新候选结果显式保存297/267/30分母，scalar gate对null为false，原sgm.valid及science_entry保留失败，资格/complete和public summary均能完成。新增人工全流程将一个未决候选经compare_candidate、真实原efficiency_target/science_entry分支、回执验证和summary到公开聚合，未把数值门失败当执行异常或自动修复理由。
- PARENT_COMMIT固定为 `BE39CFB5F84C079753D3BD250EFC94E8A6177ED5`，真实archival_verified.json的local/remote都必须等于它，ordinary_push_verified严格is True。人工案例拒绝相等但错误的commit、数值1及false，不以文件存在替代依赖核验。
- source_closure已包含四个漏项，并扩展到实际评分、分类、归一化及原数值helper；overlay、manifest和verify_prepared均使用该闭包，配合原sgm.code_identity核对。原科学target、固定集合/T、独立p反推及父loader来源不再漏绑。
- category/window_settled完整变化和identity布尔值留classification_changes诊断；classification_gates仅保留冻结事件集合相等，额外coarse category全等硬门已移除。人工案例在fine集合及其他门相同时允许仅coarse category/window变化、同时完整报告ID。原主细资格、新旧细子步/COI/common/p误差、完整ID及T_base门仍独立保留，不因此放宽原科学准入。

### 6.2 剩余核查与范围

已复读实际sgm/group_summary原helper，修订所取official_horizon_risks确有稳定1/2/5/8s value/coarse_value定义，未把别的统计量误作risk或RMSE。各主指标的u_score继续用自身fine/coarse差，T_base及原T_anchor逻辑不变；完整结果保存candidate_metric_reproduction供最终有值指标的对应核查。合法null路径获得完整负结果不等于其指标通过。

主入口仍在所有函数之后；prepare单writer冻结基线与候选入口，evaluate各候选独占目录且正常至多一次sgm.evaluate，已完成核验后复用、已started未完成拒绝自动重放。补丁的严格父加载、数学来源、overlay往返、原对象不变及deepcopy不丢p_backward检查未改变。没有新增第三候选、参数拟合、训练或FOM。

当前建议代码retain_branch（修订实现已审）、模型retain_evidence/no_change、科学continue当前正常数值资格。真实执行后仍须核两候选全部门、数值未决/事件分母、成本与回执，通用实现采用还需两者全部合格及独立结果审查；原科学不合格的A16不因数值资格通过而晋升，旧M3不解锁。
