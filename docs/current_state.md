# 当前研究状态

任务BLOCKED（新执行范围待授权），cwd E:/03-nn_Transfer_Func/kiki。人类已批准的一次QA05实际完成：135项收集、133通过、2失败、无设置报错。原26项超长路径错误消除；整体未获资格。正式训练/更新0、合成val未生成，无活动科学PID、队列或锁，未合main、默认模型不变。QA05授权已用完，不能重复或擅启QA06。

**决定性证据**：实际源5b7722c0cf66b3eb5313d9f4efd6d2f007a551e4，worker44068/FILETIME134371321358411994；pytest75.83秒，native wall80.681267、CPU user10.484375/kernel19.3125，exit2、Job19累计/最终0、cleanup=true、外层无遗留清理、106源前后同、锁清。native SHA34283382ab6da9e5901062986bdfacd3301a1379cf3d423ef576e0fb3f4dd548。[QA05结论](../reports/qa05_recheck_20261022_conclusion.md)、[摘要](../results/rational_execution_20261021/qa05_summary.json)。

**两失败**：历史QA04正例测试用大写QA04/receipt匹配小写native_qa04/receipt.json，根因明确，修正只通过AST未复验；该测试后续正例当次未执行。正常父子均exit0但所属Job仍有活动成员，严格清理并判FAILED，子Job累计3/最终0、无成员映像记录，不能声称conhost或纯计数延迟。原数学8、runtime16、Job七项中六项通过。

**具体待审修复**：实际gpt-6-astra关闭评估/root决定在astra6_after_qa05_20261022.md。静态diff qa05_followup_prepared.diff已准备未应用：pytest/父子派生每次显式DETACHED，不混NO_WINDOW/不breakaway；从已核自有Job句柄记录活动成员PID/FILETIME/映像/wait，立即按旧严格规则清理，无宽限、白名单或放行更改。diff基于QA05保存源，其中正则hunk已单独在canonical测试修正，不可盲apply。内存AST通过，待非实现者短范围复查。canonical A2 Job/helper/probe未改；应用后须新源重新取资格，不套用旧A2。

**下一动作与权限**：仅可完成安全静态准备/选择性归档。拟请求新增执行范围：固定五例normal父子、父子孙、输出管道、真正孤儿、监督进程退出，全部符合预期再完整四套QA一次，通过自动原12模型；任一失败保存结束，不修后重复或换名续跑。新增执行尚未获授权，不把“取消进程额度”解释为取消验收次数。旧“允许复验，通过后继续训练”已履行唯一QA05，不重复询问同一已获训练许可。

**科学边界**：原0_rational→0_tanh→1_rational→1_tanh，各seed6301/6302/6303、各512，共6144；0/32/128/512保存、32真实恢复、最多一次非择优中断恢复；12端点冻结后一次合成val。原项目val/test关闭、TASK4暂停，40状态/100ms一次初始化/真实区间/失稳样本/数值与晋升门槛保持，旧18×5126085.6472446、384测时不重跑，数据条件配额/24valguard限制不变。执行器失败不评价模型族，父L1/L2未完成。

**保留与恢复**：原QA01–05分类/来源/费用全部保留，qa05/failed_source保存失败core/测试；A2旧15依赖资格仅覆盖旧源。入口private_data/closure_mechanism_20261019/P1/rational_execution01/{make_binding.py,launch.py}，q5与qa05输出保留，禁止pytest清理。全部进程额度null、wall/CPU照记，旧hold/REJECT不改，支持成本仅报告已知部分。归档分支codex/edge-memory-discrimination-20261021，本次同步核验进行中；main仍ad1b48a3b701cfaf951d0e96067d5fb537e74c90，local HEAD/index保持，不reset/强推/删证据。heartbeat保持，待新增权限时安静，无重复诊断或队列。