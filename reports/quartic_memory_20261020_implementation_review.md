# 正四次记忆储能：独立实施审查

2026-10-20，独立代理`/root/fixed_edge_final_review`。未参与提案或实现。本次只读协议/方案挑战、新runtime/runner、三个测试、新BASE的launch/make/QA driver及直接旧接口；未执行程序、测试、模型、测时或统计，未读源权重、原始数据、val/test，未递归委派。仅新增本文，不改源码、协议或账本。

**当前有两项确定的冻结前阻断，root已同意一次纠错；纠正并冻结实际源码/命令后，其他已读范围可进入登记的原生QA。** 此意见不是QA通过、实际来源等价或完整成本准入。root明确本次纠错bundle消耗全轮唯一前置工程修复：之后QA01若出现新失败，保留关闭，不再修复或运行QA02。预算上限保持，不能将预留误作重试授权。

## 必须关闭的两项

1. `BASE/make_invocation.py:51`与`BASE/launch_stage.py:83`引用不存在的`reports/astra6_quartic_memory_20261020.md`，实际原案为`reports/astra6_after_spatial_extrema_20261020.md`。前者会在生成绑定时失败；仅改make不足，因为launcher还要求旧错误key。两处须统一绑定真实原案，不复制或改名制造原案。本次尚未生成invocation或启动QA，root已先行发现并采纳修正。
2. `experiments/quartic_memory_20261020.py:113–138`的check_endpoint按路径/metadata判角色，但N没有验证完整七项，Q仅确认存在quartic项。含Q的JSON若标N可通过该层，随后QuarticMemoryRuntime又仅验证N/Q集合之一，不能替代角色检查。冻结与评价消费者须以KEYS[v]核完整集合、长度与无重复，系数长度/全部有限及双副本一致，再检查Q的非负a；添加错误N/Q termset的合成拒绝。root与本代理独立读到此缺口，已采纳最小补强。现有new_stage按角色构造正确；这不是已有科学结果问题。

纠错前已读源码身份列于文末。修订后root须保存新hash/准备费用及新固定命令，不沿用本文旧hash假称一致。本文封版保留原发现，root负责核上述明确diff及冻结。

## 数学、优化与缓存

最终runtime的node q⁴直接使用当前q，新增gq=4a q³后进入原J/R；原能量/梯度、端口与物理单位沿用通用编译器。a位于coefficients中，通过term_key查找；_check允许完整N/Q集合的重排且拒缺项/重复，不以末索引猜系数。make_optimizer将该向量归closure。build_variant在私有副本中保留原七项参数与权限，Q精确零a，不改来源实例。

_products第76行缓存的是仍连接Parameter计算图的a张量；field第130行只按有无该AST分支，不按a==0跳过，也没有detach。每次graph结束/异常、加载state、设备迁移或更改terms都清除缓存。投影/导出拒绝活动graph；更新后重建缓存，不应持有旧a值。缓存测试定义实际检查零a的grad_fn和更新后新对象，不是只查字符串。

one_update逐项核原目标/组件、raw梯度、clip范数、Adam后参数/矩有限，再只投影a并保留矩，记录raw梯度、投影前后参数、Adam与投影增量。负无穷不能被clamp掩盖。原目标直接调用closure_training_runtime.objective，原long/local/hidden/torque和100ms接口没有改写；没有新增L1或空间损失。

QA定义正确区分N/Q零a公共raw梯度与fastQ/genericQ同参数集合的裁剪/Adam比较，没有要求新增参数后的N/Q公共更新相等。包括非零能量/梯度、NumPy场、原点梯度/Hessian、a零点未锁死、term重排/closure组/只投影a、矩保持、双系数副本、负a导出拒绝、非空Adam恢复及cache错误清理。机器重编号测试核新增项本身的置换性质，不应称整个物理网络重编号路径已额外重跑。CPU/CUDA分支定义存在，实际执行及设备资格仍待QA。

## 生产来源、日程与恢复合同

三个来源显式绑定空间轮manifest及O512三文件hash，6101保留temporal原路径，6102/3保留spatial O原路径。source_metadata核step/seed/O/1280来源；prepare核完整385身份，从各源NumPy RNG/coverage生成新的1024日程并写固定prepare目录。new_stage在模型构造及空Adam后恢复源CPU/CUDA/NumPy RNG和coverage，train逐行核采样、前后RNG及long/local真实计数。force保持1.0068379869405584，1/2/5循环342/341/341，apply_lrs更新513起乘0.3且不重建Adam。

正式入口拒512/768、旧角色、resume/external-recovery/extension；train内残留恢复分支被前置validate_request挡住，不能据其存在获取续跑许可。每32保存加stage0，共33个checkpoint。save_checked和export要求投影后有限非负a；restore_stage先核导出双副本，再调用原原子身份恢复、加载Adam/RNG/coverage，最后原Parameter身份上启用新runtime。预检模型构造消耗的随机数在原restore_checkpoint末尾被恢复覆盖，非空下一更新比较仍需实际QA/cost证明。

runtime的from_export核terms/state及系数双副本；CPU generic/NumPy表达式继续含pow4。cost/evaluate都通过新runtime，不调用只支持旧七项的linear_cache。评价只认本轮1024角色，要求六个冻结条目逐项hash一致后才推进；freeze_endpoints仍为显式函数，root须登记唯一调用、费用和远端冻结身份，不能把函数存在写成冻结已经完成。

## 成本、监督与评价范围

cost各组实际12次：1s长面板后保存非空Adam，下一真实5s长面板参考/恢复匹配各占已有一次，不添加优化更新。固定长/短面板与CPU16身份在prepare冻结后复核，六来源零更新比较保留/恢复外层CPU/CUDA RNG。估价使用342/341/341、33次保存、加载和20秒监督开销后乘1.25，每模型≤3300、六模型≤19800、每评价≤340才allow1024；train实际消费该flat admission，不能降档。抽样保存/测时仍是估算，不保证正式1024在硬cap内完成，失败按原无重试规则保留。

私有helper的production消费值为QA900/870/840、prepare200/170/140、cost1800/1770/1740、train3500/3470/3400、evaluate400/370/340（native/PS/self），均满足实际PS硬3600。make对train显式传3400，其余cap−60；launcher实际传cap−30。静态execution_contract不证明尚未生成的全部实际argv，root须逐类对照生成命令、parser与PS并冻结后启动。

完整prior集合、未结清owned清理拒新launch、独占lock、真实PID/FILETIME、两层exit/complete/GONE与源前后hash沿用现有合同。prepare/cost单次，正式重复角色和恢复被拒。implementation_cost.json已存在，qa_seconds26.1505654=8.3153334+17.8352320，make纳入、command_gate要求，prior同时扣QA4200和含支持3800的阶段33000；纠错费用仍须补入同一累计文件，不转移或重复计嵌套子费。33000各类预算没有用P3/P4或repair填补。

binding拒整体.h5/.hdf5，make只取已登记train/物理叶身份、源O三文件和直接代码依赖，不递归发现新数据文件。load_events限旧train+精确新train目录，完整385/342/43及源metadata身份均保留，不访问val/test。cohort按固定366/19身份，不按ID前缀。

8s与真实区间至15s资格分开，actual15完整联合real15及pair15的有限/分类一致/原阈值；空间解释仅在完整actual15资格下赋值，否则null。推理异常保留整批占位及每事件exception，未删事件。现有raw summary不应作为另一个绕资格总分；正式JSON提取必须沿用相关资格失败总体null的原口径，不能用finite子集或15s资格救8s，也不能将新19称独立确认。局部谱/曲率仍不构成非线性区域证书或采用决定。

## 审查时身份与结论

| 文件 | SHA256 |
|---|---|
| runtime | 53af6d143c66ba86b1f0cb503e5a791bf604218acf8d388e7ed098fb583c1a68 |
| runtime test | b402eed5ae23bbc652a1d6d20b9f72047ec8da8cb52ae6590ceba32f5c31dfea |
| runner（纠错前） | 87f5b8a9f676e6e02ec2ae16d57d4d511583e7e48c2a596365f17c0881fbe295 |
| runner test（纠错前） | 8980c0764712c306c121bbbbd5b0b67db8d128be14cfef02bdd4bfd002bc01ff |
| native test | f38317f78cd3a5b30e77635f53b1cb9f8b498905dd510a4a6bd75433e7fe8e42 |
| launch（纠错前） | 012d42c4e8e83ad7917bfe5573be429b8d1003c1d92e786bd5fe49f6dc40552d |
| make（纠错前） | a4762fac89a2dbace0fc425ab8690410825d9fa4efc026a37ede08fb9665b9e7 |
| QA driver | 3042f657b448d3ccc3ab23cbc812fb7f8096971737f34e310c8bc8349ca2772c |

除两项已列纠错外，已读范围未发现另一个确定数学或生产接口阻断。纠错、累计准备费用及实际命令冻结完成后，可按当前收紧后的唯一QA01验证；不得将本静态意见称测试通过、完整成本可行或模型候选接受。工程PR最多选择通过实际检查的runtime/对应合成测试/必要说明，不整体合并实验runner/helper或默认端点。

本次一次支持审查采用最多600秒保守份额，嵌既定3800hold，不新建原生任务、不扩预算或重复计费。本文写毕停止编辑，root核纠错与新hash、保留最终科学决定。
