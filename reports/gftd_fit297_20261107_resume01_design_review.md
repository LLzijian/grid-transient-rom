# Run3 首次I/O中断：取证与无重放续行设计审查

结论：接受“保留两模型及41项初始化证据、只继续553个未开始配对”的技术续行设计，状态 `continuation_design_accepted_implementation_pending`。没有发现已完成初始化输出或诊断丢失，不需等价输出代替，更不能重fit或重试前33项数值资格失败。新resume/test尚须冻结实现审查和root单独释放；本报告不执行续行。

## 原证据绑定

| 文件 | SHA256 |
|---|---|
| `results/gftd_fit297_20261107/execution_release.json` | `F10A023FAF1783DB600230C0AB370CB9F919E73DBEAF08B470A0B9C0BD160443` |
| 同目录 `execution.partial.json` | `98C570DE7F466C2D7397FAF021F8433CC7BE7AC7070C1BD866C77A5A48F719D9` |
| 同目录 `execution.partial.json.tmp` | `F32FFB1AACE07274AB3B3F38DD0FEA6527A1D1AD71533DF0C6B8D1EE6F4C20B5` |
| 同目录 `execution.stderr.txt` | `7093C56B51900CFA2B4512ABC107DE4106C7F123A7FA62DE464FF09644940C2F` |
| `private_data/gftd_fit297_20261107/static_model.json` | `97714CFED13B4E243924D87A6DE7590F76BD7FDED68C4322D22D4F907FA4C032` |
| 同目录 `modulated_model.json` | `86ADEF0B5DB08F56594EBBD4B78924E6B9FB43E7AA2C47F1D8FC91BAEC1EA7A5` |

原runner仍为 `7E680ABF096D2296903BBB4ACE52A6018C27C3F7551874A22E1249B1D13DFFC4`；原release的51源当前全部匹配。原始partial/tmp及所有原文件保留字节，不原地“修正”旧状态。唯一进程session35491在UTC 2026-11-07 17:06:59退出1，无重试。

本次只读回溯、JSON及文件字节哈希，求和保存计数；没有数值函数、测试、拟合、积分、谱或事件载荷重读。

## 首次失败位置与可恢复边界

回溯首先指向原runner **386行**：`receipt['artifacts'].append(detail); persist()`。`persist→write_json→atomic_bytes`在 **75行 os.replace** 将完整tmp改名为execution.partial时抛WinError5。随后finally再次persist，因 **71行临时文件已存在**抛FileExistsError，使最终execution.json未写出。第二错误是第一错误的后续，不是第二次科学执行。瞬时读句柄等原因仅是假设，目前根因unknown。

首次失败在static零基40、`expansion_0043`的初始化诊断落盘后。独立验证：

- `static_000.json…static_040.json`连续41项，其event_id逐一等于原冻结legal_ids前41项；没有第42项开始的证据。下一项为 **static_041 / expansion_0044**。
- 41项中8个completed，final_numerical均true，其嵌套NPZ状态文件哈希全部匹配；33个blocked_by_stage1，保存stage1_numerical.passed均false，保留为原实际结果。
- tmp是完成41项后的完整JSON，48个顶层artifact引用哈希全匹配。原partial仅47个引用，保存的是第41项attempt计数已经加1、计算成本尚未合入的前快照；不能仅凭attempt41把它当完成41项。
- 逐41诊断求和**精确等于tmp**：49优化任务、9347残差调用、9593积分、396020 RK4尝试/完成；8成功、33阻断。partial仅48任务/9130调用/9370积分/386800步。差额为第41项真实已执行1任务/217调用/223积分/9220步，不需推测或重复运行。

两fit均有完整模型和private_details。原fit合计384任务/76351调用；与已执行初始化合计433任务/85698调用。tmp的计数与持久逐项证据一致，可以生成新的重建中断回执，但不得覆盖原RUNNING快照或伪称其是原进程成功写出的最终回执。整体程序wall未成功持久化，宿主总时长应另列，不能用阶段耗时之和冒充。

原 `all_initializations_frozen.json`尚不存在，future/prediction、score、coverage及model-only分析均未开始。原进程内states列表可由8个NPZ和33个明确失败None重建，不存在未知的第42项输出。

## 接受续行的必要条件

1. **固定原证据与选择。** 新release绑定原release、快照清单、partial/tmp、两模型、输入/排序及逐项产物哈希。只加载冻结checkpoint/config，不调用fit、prepare或代理，不更新参数、尺度、门或阈值。保留41项原状态/失败/诊断及成本，原41不能再次initialize。
2. **只继续未开始配对。** static剩256项＋modulated全部297项，共553。每项仍为原simple600→smooth1200，阻断即关闭其后续机会，不借技术修复扩预算；最多新增1106优化任务/995400残差调用。代理余额和已阻断stage2机会不转移。全594配对和297/267/30原分母保留；完成后先冻结全部初始化，再原双步未来及L1评分。
3. **丢失缓存只影响辅助覆盖信息。** `_CACHE`未持久化，原prepare不可重做594次，更不能重跑初始NNLS。接受原 `coverage_information` 的cache-unavailable→unknown、零新分析成本出口：保留已经冻结的24包适配信息，明确“部署后全体条件信息因中断丢失准备缓存而未完成”，不称实测低信息、不以None填0，也不回选模型。初始化、未来、L1及L2/L3不依赖此缓存，可以继续。
4. **恢复累计分析账，不重复相加。** 两模型preparation_key须相同；最后modulated fit_state保存的是累计ledger：416信息SVD、315列投影、594记忆SVD，等于两fit实际计数之和。恢复这个累计值，不再加static的累计快照。对应未用1462/270/3406保留关闭/跳过原因，不记为已运行；不能新建零ledger抹掉已用成本。
5. **仅I/O修复。** 新独占claim/宿主回执、不可变小进度记录和新final路径保留旧字节；避免共享大partial覆盖路径，或将纯I/O提交重试严格限定为同一已算字节，不能重试包含initialize的事件循环。先登记已完成成本再I/O，残留tmp不能导致再次计算。新中断的实际诊断/成本若缺失则unknown，不伪造。
6. **原资格与范围不变。** 未来从冻结实际返回点开始；两模型的原来源/环境、主细/分类规则、L2严格证书、既有FOM/Kron身份及有限次数继续绑定。辅助coverage unknown不阻断这些主步骤，也不构成模型采用依据。最终报告区分原进程、技术续行、失败样本、跳过分析和实际成本。

代码决定：可实施这一有界技术修复；模型决定：不晋升；科学决定：保留已完成证据及原失败，继续尚未开始的原登记工作，无新增科学Run或额外研究门。本报告至此停写，待新resume实现独立审查。
