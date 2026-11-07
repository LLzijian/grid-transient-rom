# Run3 冻结引擎独立实现审查

当前结论：主体符合最终协议；有一项细格覆盖分析异常出口需修，状态 `pending_fine_information_exception_fix`，不接受本v1执行。API/runner仍待整体审查。本次只读冻结源码/测试及父归档差异，未读fit297事件、运行真实fit/优化、积分或新科学条件。

## 输入绑定

| 文件 | SHA256 |
|---|---|
| `src/kiki/models/gftd_fit297.py` v1 | `D82983F28D36A7570C03A971EF4142E6C77910EC250D629C0F4063F782371DA8` |
| `src/kiki/models/gftd_fitting.py` | `57EFFAAB858200BF77A33DE6661210104BB01BAE2DEF4E8DB718C08254D982F4` |
| `src/kiki/physics/kuramoto_functional.py` | `5A709BC2B5B984FBDD4C376F69852E0A53A12929DEE3F525A453B5934CAEC021` |
| `tests/test_gftd_fit297_engine.py` v1 | `04D9F9BCC4B1ED424C6C2CF89F31B28C2B4E490739106AA16D3A8B0E1FDC54EF` |

依据协议 SHA `CA22743C53FE4FE398960E111838A373F8C4686EAB0974C5DAF6D9C28435062F`，父Run2归档 `66f1308c25ee38c3d7217bd86ee43b61a042a5b6`。父方报告95项新/旧回归通过，本次未重跑；这些是确定性工程资格，不是实际拟合充分性证据。

## 必要最小修复

`coverage_information` 已将主格重积分的 ValueError/RuntimeError 等转为带已用counts的unknown，但细格重积分发生在 `_audit → _candidate_systems`。v1 `_audit` 只捕获 `_AnalysisLimit/FloatingPointError/OverflowError/LinAlgError`，遗漏 ValueError/RuntimeError。核心状态/非有限校验可能抛ValueError，因此细格失败仍能逃出report-only覆盖分析并中断action。

最小修复是在 `_audit` 对这两类异常同样返回unknown（或在coverage调用处等价保护），保留现有局部grids和已用积分/RK4/information_history_steps，不能重试或丢成本。新增主格成功、细格ValueError/RuntimeError的定向mock，核unknown及计数。root已保全v1并安排此项，不需改科学门、预算或重跑无关资格。

## 其余固定逻辑核查

- **严格配置/尺度：** 五键观测验证后，新request严格白名单、固定均匀整数索引、正节点尺度及k_ref、来源hash；σ0限定。新格式完整指纹及旧格式分派明确，不替换旧协议常量。r=rstar*u、a=.1*u，simple/smooth与代理起点一致；新逐机历史残差保留完整ω，原/.1数值资格另算，未随训练尺度改门。
- **一次全体准备与子集：** 缓存键绑定有序包、固定非边物理、request及网格，排除待拟合k/边门；两个方法同key。全体两格逐包缓存Z分解，只做一次初始static筛选。`project_cached`每个范围用其自身总行数及共同记忆奇异值阈值，未把全体归一化矩阵截行为24范围。准备成功深复制复用；失败按key固定原成本，第二模型返回cached failure和零新成本，不再次读/重建/筛选。
- **候选与选择：** 只用24包候选完整ODE的a建立精确指数列，参数变动重推；实际[u,v]字典与解释[k,β]差列分开。k_ref先验/阈值、一次去偏、删除层级及必要NNLS失败传播继承父规则。初始＋三提案，拒绝不延续拒绝状态；接受只按新参数重新推断后的事件等权history目标。弱目标下降不替代实际目标，未来/标签没有入口。
- **信息和预算：** 单事件只一次谱/列范数，不逐列投影、不作全列恢复门；共享完整列才审条件信息与跨格完整秩。global ledger对信息SVD/列投影/记忆SVD实际调用先计数，1878/585/4000封顶，超限unknown不加额度。缓存Z分解只做全体两格，明显在保守4000内；每个提案不重新做完整信息审计。部署分析只用给出的实际成功state，失败None不补a=0，不回写参数。报告须把两范围、ID/数量与raw/null分开。
- **完整初始化及旧兼容：** 旧引擎新增可选新配置的endpoint/residual委派，旧调用参数默认路径保留；initialize新格式先严格验证再进入同一simple600→smooth1200流程。rstar变量在stage1→stage2可行点传递中一致，p硬约束，实际返回点自身原尺度数值检查、同积分步损失守卫及typed失败保留。内部200×2代理不冒充部署初始化。新预算仍受真实残差计数（含Jac/初值/守卫）限制。
- **物理核心：** 相对父归档只有预登记的幅值梯度向量化和关闭状态门时直接返回常数耗散。向量式为 `gamma*(abs(B)@(K*s_mod*exp(h)*psi))`，与逐边幅值偏导同式；完整端口、相角力、储能及启用门的旧分支保留，shape/finite校验仍在。允许浮点末位变化，不声称逐位相同或实际轨迹已获证。
- **审计产物：** 引擎返回完整private_details及hash，公开fit_state压缩大历史/矩阵；最终API/runner必须真实持久保存该私有证据并绑定hash，不能只存其摘要而失去独立复算依据。失败原成本、双方共享preparation归属及各analysis ledger同样在整体验收核对。

修订后只需核定向diff/测试，不增加研究条件。模型不晋升，联合可辨和优化充分性仍可unknown；最终科学执行须完整实现审查和root释放。

## 最小修复最终复核

修订 `gftd_fit297.py` SHA256 `43F937B6973490C771AEA2E1AC8694A2C356A48FB2F26337A73C9E991C6E334B`；测试 SHA256 `8D331D31C5AED79D44150BC01839CD48CC8C776E4264E9D7ACC79B9C6CB1E9A6`。独立diff确认生产仅在 `_audit` 的异常元组补ValueError/RuntimeError，定向参数化mock保留主格40步及细格部分7尝试/6完成，返回unknown，累计47尝试/46完成且无重试；原源码/测试已原字节保存在 `pre_review_engine_v1/`。

父方报告17项定向资格通过（19.69秒），原95项资格保留历史，并非全部重跑。本次未重跑。唯一组件阻断已闭合，最终状态 `engine_component_accepted_integration_pending`；其余三项绑定源不受该修复影响。完整Run3仍待API/runner/L1的集成验收、失败成本和release绑定核查，不据此启动科学执行。报告至此停写。
