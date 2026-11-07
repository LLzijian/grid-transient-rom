# Run2 repair01 独立技术恢复审查

当前结论：恢复原则及冻结候选主体成立；版本一致检查尚缺，状态为 `pending_dependency_check`，本段不接受执行。该技术修复沿原 Run、原选择和原预算，不是新增科学 Run。

## 绑定输入

| 输入 | SHA256 |
|---|---|
| `experiments/gftd_modulation_20261107_resume01.py`，v1 | `31FBDD1E6B2D1EA1A5FF5C53AB6FC1CBCC96768DD1051FF10D448E450CC8E277` |
| `tests/test_gftd_modulation_resume01.py`，v1 | `910832BE92F58198B0899694F94EB070A2C6BB3511E6C0820191D32834A15342` |
| `pre_repair01/manifest.json` | `C01EEE331F7A694D283B689BA1276AC0BAA76DDD0B0526E03E3015905F52425A` |
| `io_failure_identity_audit.json` | `FAF652029DB757A21EE04E61A8F397E749460C819EA568DB79B444E067F9078E` |

后两文件位于 `results/gftd_modulation_20261107/`。同时读取 root 的 `repair01_adoption.json`、原失败回执、38条完整 partial、fits/模型身份与相同输入。本次只读和纯保存数据字节/计数核验，没有调用 runner、initialize、fit、积分或采样。父方报告12项mock测试通过（6.17秒、exit0），本次未重跑，不把mock当科学证据。

## 已独立核实

- manifest的49个文件当前哈希全部匹配，原42源及冻结选择保持。16 fits、38条持久记录、48计划配对与原回执39次完成一致，summary尚未产生。
- 序号按零起：36/37/38均为batch6、static、同一个冻结模型 `9A34EB9280FF70504688D14BEE4873A390DAE32D696CD2B10D35705E21BFE431`，分别obs_018/020/022。三条四键输入形状与little-endian float64字节完全一致，含signed zero；配置由同一模型文件绑定。36/37的state和双步future也逐字节相同、有限，原final_numerical均通过。
- 失败回执减durable partial的差额恰为1次初始化尝试/完成，2优化任务、21残差调用、27初始化积分/1380 RK4步，以及2外评积分/300步；拟合、生成、A资格等差额全0。此为实际已执行成本，不是推测或重复运行的成本。
- `recover_lost`只为index38复用已认证输出和同点资格；stage1/stage2、实际优化trace、返回点来源、耗时保持null/unknown，资格中去除来源执行的counts/wall_seconds。复用不恢复原第39项实际诊断，也不增加一次初始化成本。若余9条均持久化，真实初始化诊断覆盖为47/48，而输出覆盖可为48/48。
- 继续循环仅index39…47九项，加载原模型并校验参数/RHS；不重fit、不重放丢失项、不生成噪声。累计计数从失败回执39次开始，包含lost成本。typed失败仍保留对应行和原分母。
- 小不可变事件文件经临时文件、flush/fsync、replace写入；独占标记/临时文件阻止重复运行。48条最终输出和哈希先冻结，才调用原validate_counts/score。保留原failedreceipt，另写恢复回执；最终检查原证据、新源及release未变。不覆盖已有partial、fits或原回执。

## 唯一必要修复

`load_context`须在started及任何继续计算之前，明确检查当前NumPy/SciPy版本与原失败回执一致；缺字段或不一致立即拒绝。原回执记录 `numpy=2.5.2, scipy=1.18.0`，也是原runner固定版本。原42源码未变不能证明运行依赖未变；此检查属于原恢复身份条件，不是新增科学门。加入版本不符/缺字段时无started、无initialize的mock回归，待新hash作最小diff复核。

没有发现其他需要修改科学算法、目标、预算或重做fits的阻断。I/O根因仍unknown，不以本修复伪称已查明。最终接受意见待上述修订冻结后追加；本审查不预判机制、预测或模型晋升。

## 最小修复复核与最终接受

最终恢复脚本 SHA256 `9A1E2350E09104B1B85254029F288978760473CF6F12DBAF624479A39B0C0F11`，测试 `7B994B92B650F230C96BB8EB4B488AA42BC7210FFFEF126683694BB66C7C225A`；资格记录 `repair01_qualification_final.json` 为 `C432810C36A524015D3CC85D3F1BD60C63FD05FC14921274D4E8C2EEA425A588`。原v1源码/测试保存在 `pre_release_resume01_v1/`，哈希与本报告v1相符。

独立diff确认生产代码只新增四行依赖版本检查：原失败回执须有冻结NumPy2.5.2/SciPy1.18.0，当前运行版本须相等；检查位于load_context内，在独占started与继续计算之前。测试仅补版本元数据并覆盖当前NumPy/SciPy错误、原回执缺字段/版本错误五类拒绝，均要求无started且无初始化。父方报告17项mock资格通过（12.20秒、exit0），本次未重跑。再次核验原49文件哈希无变化。

唯一阻断已闭合，最终状态 `eligible_for_root_repair01_release`：允许root释放**仅剩余9项**初始化及原48项冻结后评分。原16 fits、原预算和选择不变，index38仅等价输出复用，其实际诊断仍unknown；原失败回执不改。恢复结果仍须独立复核，接受本代码不等于已完成恢复、机制确认或模型晋升。报告至此停写。
