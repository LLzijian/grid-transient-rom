# 实际Astra6新轮次原始方案与root采纳

模型gpt-6-astra，agent /root/astra6_qa05_plan_20261022，新授权之后实际followup规划；以下保存原始返回。

本次为指定 **gpt-6-astra（Astra6）新轮次只读规划**。已核对准备补丁、独立复查及现有运行环境选择验证代码；未修改、执行测试/probe/训练或操作Git。

**建议采用已获人类授权的固定路线：实施补丁 → 五例检查 → 一次完整QA06 → 通过后原12模型训练。** 不再等待同一执行许可。更新提案中已过时的 `NOT_AUTHORIZED_EXECUTION`，保留旧版本作为历史。

最容易遗漏的是以下连接。

1. **先保存旧A2，再冻结本轮源码。** 保存旧Job/probe/helper、旧selection原件及其hash、旧A2观察目录；旧selection里的绝对路径随后可能指向已修改源码，因此它是历史证据，不能继续作为当前合格入口。新五例和QA06使用独立目录，不能覆盖旧A2或QA05。正则已单独修正，不盲目应用整个diff。

2. **五例启动不能依赖尚未生成的新selection。** 这会形成资格循环。沿用已有低层 `python_command(..., "direct_base_venv")` 和 `command_environment` 构造五例命令，由新授权、固定源码清单与原生Job监督执行；仅在五例全部符合预期后生成新selection。`selected_python_command`继续用于完成资格后的QA06及训练，不跳过其验证。

3. **现有selection格式可复用，但必须补上“五例全合格”准入。**
   保留现有 `normal_receipt`、`orphan_receipt`、`baseline_reference`、`parity_evidence` 字段，分别指向新父子正常例、新孤儿例、原已保存公共参考、新正常例metadata及真实孩子身份。旧参考只作逐字段比较，不重新执行。新selection的 `observation_source_commit` 必须是本轮冻结源；`observation_attempt=1`可表示**新授权批次**的首次观察，不能解释成旧A2追加次数，独立目录和授权绑定必须明确这一点。

   **现有helper只核normal/orphan，额外放入三例字段不会自动获得验证。** 最小新接口是在核心增加一个固定的 `validate_descendant_batch(binding, commit)`：核验五例清单、实际回执、预定角色、源hash与身份结束证据；QA06授权验证和正式准入均调用它。无需放宽或通用化原selection schema。新selection的 `files` 应包含五例清单及所有被引用证据；全部 `.py` 项必须在normal/orphan的 `source_before` 中存在且相同，因为当前helper逐个检查这一点。

4. **选择路径必须贯穿实际调用链。** 新selection采用明确的新路径；同步核心 `RUNTIME_SELECTION`、binding生成、launcher、worker receipt、正式恢复，以及测试里硬编码的 `SELECTION`。历史QA04/05验证函数保留历史语义；它们的fixture可继续隔离构造，不能要求当前新版源码重新满足旧A2 hash。QA06的准入既检查新selection，也检查本轮五例资格，不能回退到旧selection或QA05通过函数。

五例按注册顺序各执行一次，最小观察方式如下：

| 案例 | 必须保留的实测证据 |
|---|---|
| 正常父子 | 父子PID/创建时间；实际创建参数；子已退出、父退出0；COMPLETED、无清理、Job0；同步metadata与14项CUDA结果 |
| 正常父子孙 | 三层身份与两次派生参数；孙由子等待，子由父等待；全部零退出、无清理、Job0 |
| 正常管道 | 使用实际stdout/stderr管道，核对固定内容；实际派生参数与身份；全部零退出、无清理、Job0 |
| 真正孤儿 | 孩子先写身份握手并阻塞，父退出0；native FAILED、实际清理、孩子GONE、Job0；新活动成员记录应包含该孩子对应身份 |
| 监督进程退出 | 内层监督进程固定非零退出，持有的Job句柄关闭；由仍存活的外层记录内层监督者、worker及后代身份结束，外层Job0；保留内层started记录与实际外层回执，不伪造内层无法写出的完成回执 |

监督进程退出例应由存活的外层核验**已握手的具体进程句柄终止**；不能仅凭PID查不到就认定清理，也不能添加“等一会儿后算正常”的宽限期。正常例不得接受 `orphans_cleaned=true`。活动成员取证异常只记录错误，原清理与FAILED判据保持不变。

四处创建标志的覆盖必须落到实际调用：五例记录其实际派生参数；最终QA06记录核心启动pytest的参数；新增单测拦截并转交真实调用或检查传入参数，覆盖测试外层 `popen`、嵌套child及probe child。**不要为了提前验证核心pytest参数而额外启动一次pytest。** 初层保持原SUSPENDED/DETACHED/Job绑定顺序；后续每次Python派生均显式DETACHED，不叠加NO_WINDOW、NEW_CONSOLE或BREAKAWAY。

QA06新授权与协议应固定完整QA05失败回执SHA、源 `5b7722c0cf66b3eb5313d9f4efd6d2f007a551e4` 和FAILED状态；`34283382...`只能作交流简称，代码必须使用实际完整SHA。三个入口的解析、历史身份循环、输出目录、worker身份、结果摘要、绑定生成及 `run/evaluate/resume` 全部核对，成功QA06才是当前放行依据；拒绝QA07，QA06不可恢复或覆盖。尤其不能遗留 `qa_attempt in (4,5)`、旧 `==5` 分支或正式阶段对QA05完成文件的引用。

停止条件固定：五例任一不符合预定结果即封存该批次，不修后重跑，也不执行QA06；五例全部合格后QA06仅一次，任一失败即保留失败与新源，不追加QA07。全部通过后自动继续原四组×三seed×512更新及原恢复、冻结和一次合成val安排。全部进程额度仍取消，科学门槛、原项目val/test关闭与TASK4暂停保持。

这是本机执行链的有限工程修复。既有QA05失败不能证明模型优劣；本轮通过也只建立执行资格。root完成独立静态核验、来源归档及最终采用决定后即可按新增授权实施。

root采纳：全部采用。旧A2原件已逐项保存qa05/retained_a2_sources；按四文件入口与Job/probe/观察器分工，先完成全部静态检查与独立复核再冻结源。五例低层启动避免选择循环，新selection与五例manifest都必须被QA06和正式准入实际核验。监督退出案例外层保留具体孩子句柄，真实Wait终止后关闭，不由PID查不到代替。核心pytest标志只能由本次唯一完整QA真实记录，不能提前另跑。新授权明确已到位，不再请求同一训练或修复许可。
