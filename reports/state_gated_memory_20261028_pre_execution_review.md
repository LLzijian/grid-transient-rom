# SGM执行前独立集成审查

审查者：实际 `gpt-6.1-sol`，独立只读角色，未负责本实现。2026-10-28。根代理保留科学采用、冻结与主分支决定权。本次只核新model/study/execution的决定性接线，复用已有纯CPU检查，不运行测试、真实事件、缓存、CUDA、optimizer或原生Job；未读取val/test或任何未来结果。报告写作按用户指定anti-defensive-writing的事实与主张边界原则组织。

**本次提交版本存在一项执行阻断，修正前不应冻结启动真实资格。** 它是评价包装接口错误，不是数学资格失败；不得因此改变精度、物理门或原历史失败分类。

## 唯一阻断及最小修正

**Q1：评价计时包装丢失新初始化器的关键字参数。** `experiments/state_gated_memory_study_20261028.py:548–555` 将 `candidate.encode` 替换成只接受 `history` 的 `timed_encode(history)`；新模型 `experiments/state_gated_memory_model_20261028.py:101–103` 的 `initialize` 始终调用 `self.encode(h, history_dt=history_dt)`。评价通过study:516–527进入这条路径，故即使使用默认步长也会抛出 `unexpected keyword argument 'history_dt'`。study:569–572随后把整块预测写成NaN；真实A/B/C零更新资格将全部未决。R_N原类的encode接口没有该关键字，因此不能用R_N评价成功替代三组接线检查。

最小修正是让计时包装完整透传 `*args, **kwargs`，仍由原encode接收实际历史步长，不改模型、来源、粗细路径或资格门。补一个零optimizer、无真实数据的接口检查，覆盖计时包装后的新初始化器分别收到 `.0025` 和 `.00125`，并保留包装前后主配置不变；无需重复11项数学套件。现有study八项纯检查覆盖 `_paired_rollouts` 自身，但没有覆盖这层完整评价包装。

## 已核决定性路径

- **模型与信息权限。** model:43–80实现A常传播、B按同状态同参数的加权门匹配总trace、C逐边门；没有正总量下限或epsilon分母。model:82–94的状态依赖耗散块与原H及反对称J相配；H不含Q，不引入额外势能梯度。model:119–170只用11点历史delta/omega和末点p，准确反推独立p流，逐子步反向推进非对易r仿射流，保留五段Simpson、F/q正则和QR完整反传。缓存仅保留参数常量，Q随状态计算；不缓存事件门或未来真值。gamma不存在，原M参数共47个。
- **粗细精度接线。** study:513–530显式选择主历史 `.0025` /参考 `.00125`，未来 `.01` /`.005`分别初始化，引用步长不改模型持久配置；异常也关闭缓存。model:187–218保存并恢复主history_dt，参考模型不能导出训练检查点。当前Q1位于这条路径外层，必须先修。
- **跨模型源资格。** study:187–265只读各一次既有R_Nfine/Afine缓存并核哈希、唯一297身份、11行真实历史、3001×40形状；双向配对保持原 `.01` 观察网格、索引10、stride2/coarse_stride2，分别检查8s和真实n、双方全部细子步有限性以及原物理/分类/评分门。execution:37–74、240–249把cross文件精确路径、哈希、身份、source、范围、方向声明和297合格行绑定到后续阶段。R_N保留原类准确历史，A用自身fine历史，不共享初始化。
- **真实区间风险。** study:343–378按每事件 `n=min(1500,len(x)-11)`，用原_point_error在真实n上分别计算细/粗误差，再以原risk聚合全部失稳样本；`unstable_actual15_risk` 和粗细扰动独立于旧最多8s字段。原分类及稳定分母保留。study:425–434的延长使用这些字段和原稳定、p、共同频率保护，不是旧字段改名。
- **确认时机及第二种子。** study:335–358复用原物理恢复和首次连续五点确认窗口末端，时间 `.01*(k-10)`；latency不可覆盖、未触发和数值未决均NA，不用标签或记录终点作时刻。study:437–462使用R_N/A/B/C共同事件T，包含同一真值、粗细均可定位且数值合格；T空则未决。E及u用同一T，`.01+u_C+u_j`仅非恶化容差。共同96要求C真实区间漏判分别少于A/B/R_N至少1，保留真实风险、稳定5/8s、p、共同频率、8s漏判和误报保护；未决/缺对照不触发第二种子。确认定义与原 `evaluate_structured_rom.py:46–70`、`evaluate_l1_latent.py:174–218`一致。
- **0端点与机制。** study:647在资格阶段记录三组零更新完整fit，C机制也在此记录；study:288–296保留0/48/96面板和共同96完整fit。study:398–417在同一C细解的 `.01` 状态及同一C参数上切换Q定义计算力率差，按rstar/秒标准化，没有不同模型轨迹直接相减，没有新增预测或诊断链。
- **零图隔离和真实更新。** model:180–185保持原非负投影，B全零前向值合法但训练普通梯度未定义。study:323–332在真实optimizer.step后立即登记completed，再投影判退化；study:285–287、481–483先保存完整模型/Adam/RNG/覆盖位置后停止B。A/C没有人为正下限；异常保存为控制退化/优化未决，非物理失败。study:678–684继续健康分支，但缺B没有完整共同96比较或第二种子。
- **恢复、来源与有限执行。** study:22–43核固定R_N三资产，不加载旧Adam；76–78为新Adam。study:117–142恢复实际SGM子类、主步长、布局、真实Adam计数及Python/NumPy/CPU/CUDA RNG；465–510实际QA为每组18连续与16保存重建后2，合格组须真实36调用。每16正式更新保存，异常attempt/completed按call_id保留；pending不伪造完成。execution:170–198、250–255以真实journal核QA，而非强制伪造108；未决结果可以诚实汇总。总1464、正式1152、QA108，修复108/重放96仅保留且正常执行禁用，不滚入旧余额；execution:293–335无自动重放或修复入口，原生receipt明确无进程额度。
- **评价权限顺序。** study:674–691先完成两种子条件及所有fit决策并写闭合记录，才加载train88并执行最终385/固定噪声报告；这些结果不进入选源、延长或第二种子。没有val/test或默认采用入口。

上述是源码接线判断，不声称真实CPU/CUDA、完整fit297、12×8成本、恢复QA已经通过。study:630–671要求真实CUDA和每组原物理数值资格；正式训练仍在实际资格及真实恢复QA之后。CPU通过不等于正式准入，预计运行成本也仍须按固定基准实际记录并由根代理落实原优化考虑要求。

## 本次证据身份

实际核SHA256：

| 文件 | SHA256 |
|---|---|
| reports/state_gated_memory_20261028_protocol.md | `0eb76eb21901ff7f06cd014d3b59157c8c3324e5defbd39062f79b5f8d458914` |
| experiments/state_gated_memory_model_20261028.py | `ad788b1c7533ee8a803b15805f4b06af2341d7ecad4df55b541a9de0618c697b` |
| experiments/state_gated_memory_study_20261028.py | `799f49a0901a6e1e2e485d8dfff3787d5882cd497da9def57cb994f90caf804e` |
| experiments/state_gated_memory_execution_20261028.py | `97960ae4eafa998185e7247d9ac066a1543a497b770dfd05d964ba6660a36cbd` |

复用 `results/state_gated_memory_20261028/model_pre_freeze_qa.json` 最后一次11/11纯CPU资格：shared与A/B/C映射全true、顶层training_admission=false；主历史.0025的独立参考无量纲r0误差A `2.0309696274799985e-13`、B `8.88005035445458e-9`、C `1.2051303339731995e-8`，原1e-7未变。旧.005失败及前三轮记录保留，原exchange六项FAIL也未改。study八项与execution六项纯检查、AST闭包70仅按既有QA证据复用，未自行重跑。

局部修订复核：待Q1最小修订及其纯接口检查完成后，仅核对应调用、证据和新SHA，再给冻结入口结论；不重开数学或全项目审计。

## Q1局部修订复核

根代理已将上述阻断版本归档到 `8a23dade95a22c55e463c66c53ff8f4b8ce419a7` 后授权最小修订。此次仅核对应接口、追加测试及QA/闭包身份，原799f版本的阻断结论和历史记录保持原文。

**Q1已解除，本次有界审查范围内没有剩余代码阻断。可由根代理归档核验并冻结新源码后进入真实资格阶段。** 这一结论不表示真实CPU/CUDA配对、fit297、12×8成本或真实恢复QA108已经通过，也不授权跳过这些步骤或直接正式训练。

study:548–550现为 `timed_encode(*args, **kwargs)`，原样调用 `original_encode(*args, **kwargs)`，保留实际历史步长，不改其余数值定义。新增 `tests/test_state_gated_memory_study_20261028.py:103–142` 进入实际 `paired_evaluate_model` 及计时包装，再经具有同一关键字接口的fixture的 `rollout→initialize→encode` 调用；外围物理评分明确为mock。断言两次初始化分别收到 `.00125` 与 `.0025`，预测有限且无exception、初始化计数为2、持久主配置仍 `.0025`，原模型没有被计时包装污染。该测试验证接口链，未冒充真实模型或物理资格。

已核追加QA的study9项/controller6项PASS及其零optimizer/真实事件/CUDA/native声明；旧8+6检查未覆盖该接口的事实仍保留。模型11项数学检查未重跑，model/test/CPUQA与protocol均保持前述SHA。对追加 `post_interface_fix_closure` 的70个源码身份逐一只读核SHA，70/70匹配、无缺失；AST PASS沿用作者记录，没有执行新AST或模型检查。新输出base仍不存在。

本次实际核SHA256：

| 文件 | SHA256 |
|---|---|
| experiments/state_gated_memory_study_20261028.py | `dcb5d2cac1462d4f71241e888ee77db34a143f2f7a80091a34b249e18e6967b9` |
| tests/test_state_gated_memory_study_20261028.py | `04bf328f03fcbe7bf183ec1f92eb18a5b6754083483f778e137dac63125f7fca` |
| results/state_gated_memory_20261028/study_pre_freeze_qa.json | `f24fe5212c0eb66f183c56fadc0088617e1eadda71dee033d2d00cddae0e0ed9` |
| results/state_gated_memory_20261028/execution_pre_freeze_qa.json | `03efdd8051aa36cdaab502db5961d43317a11bdd28e39ab18c7755f1a4764f70` |
| experiments/state_gated_memory_execution_20261028.py | `97960ae4eafa998185e7247d9ac066a1543a497b770dfd05d964ba6660a36cbd` |
| tests/test_state_gated_memory_execution_20261028.py | `be62ec4514da4a3933f6078e8e3c9f4b756db46e01758dadecd2442073378e44` |

原exchange六项FAIL、旧SGM .005精度失败与三轮记录、科学门、来源、有限账本和数据权限均未重解释。独立复核到此结束；后续真实科学结果与采用权归根代理。
