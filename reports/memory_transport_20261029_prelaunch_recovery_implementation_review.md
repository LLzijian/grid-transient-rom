# 启动前身份恢复实现：独立有界审查

2026-10-29。实际模型 `gpt-6.1-sol`，独立审查者，未负责恢复实现。只读精确diff、原锁检查覆盖及最终测试定义；另按root唯一额外授权，仅对FG保留锁的两个原完整身份各查询一次，未扫描其他进程。未运行测试、Job、源谱、数学、性能、NNLS或训练。公开报告不含实际机器PID、FILETIME或绝对私有路径。root保留释放与一次retry决定权。

**最终静态PASS：在最新前瞻方案及当前源身份下，未发现残余实现阻断。** 这不是实际guard通过、恢复metadata freeze通过或科学准入；RECOVERY_SHA仍UNKNOWN，所有可执行入口继续拒绝。必须先由root完成纯守卫、归档核源及双重释放，才可一次新metadata freeze/run。

## 精确差异及原拒绝事实保留

本地Git对象未提供c570完整对象，因此采用既存公开source rows中的原controller UTF-8原文作精确对照；其content重算SHA与raw_sha256一致，为 `bac737ba259d14f9665915a560c45332e66e8d7308038c2f805622bc8b306a59`。该版本是root已核远端的首run源，不用本地其他分支替代。

相对它，变更限checked_base、require_release、check_locks、freeze、check_binding及四个新恢复证明函数/所需元数据。fixed_evidence、parent_proof、clean_receipt、cross/QA/performance资格、completed_phase、phase_inputs、worker、launch、run/main执行主体保持原文。原科学protocol、study/model，以及weak/SGM/FG/r-memory控制器与原生helper八个限定文件，实核均与原rows字节一致；未修改旧函数或用monkeypatch替换它们。

首恢复controller `b3d66ea7bea97933d4cc8ced47ff25ac034d7418a0f2535e123ead8663aa11b1` 仍要求FG保留身份当下GONE。本次唯一额外查询实得FG supervisor=REUSED_PID、worker=GONE，原完整常量/十进制历史GONE及恢复字段全部匹配，所以该首方案会继续拒绝，不能曾称其可直接启动。root随后实际Astra返回并在任何新增guard/freeze/run前登记第二份精确锁修订，方案SHA `d648bdb659c11296d7e9af5446f04de52e18f9cee7baa42b73f7c2532768ab32`；本结论只针对修订后的实现，保留先前阻断事实。

## 两份精确保留锁的界限

| 保留证据 | lock SHA256 | recovery SHA256 |
|---|---|---|
| training_efficiency | `f9d72f0d5fccd8b5349907d69299aad543aa9b936f3e016971a7e24acd57c3e4` | `e53d9b82c8a19d5a3899cbd04682a298f16874bce27215e18d2b297e654e709a` |
| FG retained r-memory | `d764e2c272182acc6c54dda0a9edf1309c3475b1e79916305652b2adfeb123bf` | `6ebbe0897f23530709f8acf67899414285ad3c865f05f1f6da0e92b9eb7865f2` |

第一份的独立实际核对见既有identity_review，未重复查询。第二份的两身份只查过上述一次，锁SHA、原typed常量、恢复INTERRUPTED_RUNTIME/保留标记及十进制历史身份/GONE均实核一致。新 `retained_training_lock_proof` 和 `_retained_fg_r_memory_proof` 分别固定上述路径与两SHA，再核原完整身份与历史字段，只有实际GONE/REUSED_PID可接受；原样保存历史GONE与当前状态，未伪写GONE。RUNNING/LIVE/UNKNOWN、权限或查询异常全部拒绝。

REUSED_PID只证明该数字已代表不同creation FILETIME的进程、旧完整身份不存在；不操作被复用进程，不证明旧Job成本或所有子树清理。最新第二proof保留old_JobCPU=None及measured_old_job_accounting_claimed=False。没有把这一窄判据扩展到其他锁或当前owner。

## 锁覆盖与一次恢复授权

- 新check_locks按原链分别调用weak/SGM/FG的checked_base，检查这三份closed parent活动锁全部缺失；原失败study01活动锁也拒绝。保留P1的rational native_active.lock及全部子目录active.lock扫描，只有上述唯一training精确路径能进入新证明，其余存在即拒绝。FG retained r-memory以第二精确证明替代原当下GONE子条件，其余原字段/类型检查均保留并加强为固定完整recovery SHA。
- 新retry own lock仍要求真实typed worker及完全相同pid/creation FILETIME，未声明owner、身份不同或声明owner却无锁均拒绝。没有跳过旧lock目录或调用删除/kill；原helper、锁、恢复文件和旧binding不改。
- 原release SHA `077ca8767b915242c09084cbddd8ab51e6d780acd278700514517c8f2342049a` 保持，require_release再调用独立recovery_gate；UNKNOWN、缺root/独立审查/保留旧base/唯一retry授权、任何固定证据不符即拒绝。恢复文件将方案、独立身份报告和原失败SHA绑定，新binding继续绑定其实际字节及后续审查证据，不以本静态报告自行释放。
- original_prelaunch_evidence固定旧source `c5707e4fb677eb0cb64ba378e59196c5b89b58db`、旧binding `67c9a2a10baf84c3274bd6152f21cb1915de3361befa0b8e311fc03bfd103832`、首失败 `8e9e9306690a84baca480c3765595dc463ad650aa34f860e70cf73f7232ce14c`，核原exit1及所有零工作字段，并拒旧base新增ownership/native/账本/任一科学输出。没有覆盖或继续运行旧study01。
- 唯一新base为study01_prelaunch_retry1，checked_base拒原目录/其他目录；freeze在创建目录前完成条件核验，existing目录拒绝覆盖，各phase沿原独占结果/ownership/native守卫。没有自动第二次retry、递归恢复或新科学余额。仍为1848类别上限，原零消耗与先前已消耗一次plane分开保留，不重新分解、不重跑数学或复用旧976。
- 原uncapped supervise、真正exit/cleanup、科学资格、数据门、checkpoint/RNG、1848账本保持原主体，未因身份恢复放宽。旧原生回执和第一次失败不改判；新metadata及run是否通过仍未知。

## 最终guard定义只读核查

新增19个恢复测试定义覆盖：两份精确证据的允许真实状态与拒绝其余状态/异常、路径/双SHA/typed/历史字段篡改、P1其他锁/三父当前锁/原失败锁、父base委托拒绝、新own锁完整认领、原binding/零工作不变、双root授权/UNKNOWN、唯一目录/拒覆盖，以及mock五phase一次retry。它们使用临时合成文件和模拟状态，不查询进程、启动Job或消耗科学optimizer。

原14项execution定义只补新recovery gate及freeze schema的fixture接口；原plane、CPU/CUDA字节、源profile哈希、108QA、双向cross、性能向量、parent终态和五phase断言未删。study guard文件保持原SHA。当前定义已将“第二锁REUSED必须拒绝”的过时中间断言改为精确第二锁的真实REUSED保留，同时保留live/unknown/error拒绝，没有把首方案误称已通过。

本次没有执行这些测试，19+14是定义数，不是PASS计数；原27项实际guard通过及首25/2历史由既有证据保留，不替代新恢复guard。只允许root在核源后明确决定其有限纯守卫执行。

## 最终源身份

| 文件 | 实际SHA256 |
|---|---|
| experiments/memory_transport_execution_20261029.py | `2d9fbfddbddb9c6efeec1b58c31f13638389966517caf246b7702aa5907297e3` |
| tests/test_memory_transport_prelaunch_recovery_20261029.py | `5f84c3fdd029ada52554d976eb989d2ab89ccc8b1e8084c209988a266f2a4295` |
| tests/test_memory_transport_execution_20261029.py | `9612cdb7b77b07d7b14c8e8b922e0790d1c832d4c866e7bc7df166c9d74b6725` |
| tests/test_memory_transport_study_20261029.py | `0bd70750950d4a13dc0ce87bb95b055dedc1fb6e839e513afc2e555e02157e40` |

只有本报告为本次写入。原身份审查和旧失败独立保留；本恢复不是新科学轮，不改变1848或原val/test、TASK4、新FOM、模型采用权限。
