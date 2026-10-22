# Windows执行器通过验收，原12模型GPU比较开始

实际源5ffe63f5f9bd71c68e417da4759fdbd183299799。人类批准固定五例和一次完整QA06；实际gpt-6-astra规划、root采纳及非实现者复查均在执行前保存。四处Python派生显式DETACHED；活动成员取证不放宽原清理/FAILED规则。修复共同作用完成本机执行资格，不单独推断旧QA05额外成员成因。

固定五例各一次符合预期：正常父子Job2、父子孙Job3、管道Job2，均零退出/无遗留清理；真正孤儿父零退出，FAILED并实际清理孩子、Job2；监督进程固定121退出，FAILED但外层无代为清理，具体孩子由保留PROCESS句柄证实终止，Job2。全部最终Job0、源码前后同。CUDA环境及14项float64结果与原保存参考逐字段相同，未重跑参考。五例native wall合计41.5592856秒，Job CPU user4.46875/kernel3.921875秒，wall与CPU不相加。[摘要](../results/rational_execution_20261021/spawn_chain_summary.json)。

唯一QA06：188通过、0失败、0设置错误，pytest173.37秒；外层COMPLETED，三个typed退出码全0、Job累计14/最终0、无遗留清理、锁清、133源前后同。native wall181.4855868秒，Job CPU user17.140625/kernel53.53125秒；真实worker47308/FILETIME134371359995197226。native SHA a17e626097c02ba58e63f71533c6dd90bcf17685a9d8a776818b1818129468c2。[验收摘要](../results/rational_execution_20261021/qa06_summary.json)。旧QA01–05分类/费用/来源原件保留，无QA07。

相关单测覆盖：完整数学基线8、运行环境16、Job12、薄执行器152。54数值依赖字节与旧数学QA原件一致，train_group及检查点函数AST不变；保留旧135项并新增53项。独立复查发现的监督关闭因果证据、管道结果引用均在冻结前修正。[复查](spawn_chain_repair_20261022_independent_review.md)。

验收后直接启动原0_rational组，执行会话74683；随后0_tanh、1_rational、1_tanh，每组6301/6302/6303、各512更新。真实PID及恢复信息记录current_state与费用账本；不能将启动等同完成。12模型冻结后才一次合成val，原项目val/test关闭、TASK4暂停，科学标准与数据配额不变。模型采用none；本轮是可合并的执行工程进展，模型优劣待原科学实验结果。

复现（需原本机CUDA环境、保存参考及所列本地历史原件；不上传解释器/DLL/轨迹）：
1. 固定源码与authority/protocol后，STD Python调用ROUND_PATH/spawn_chain20261022/observe.py --source-commit 5ffe63f5f9bd71c68e417da4759fdbd183299799。该批次已执行，不得覆写现有目录。
2. ROUND_PATH/make_binding.py qa --qa-attempt 6 --source-commit同上，然后launch.py qa --qa-attempt 6 --binding ROUND_PATH/bindings/qa06.json --source-commit同上。唯一QA06已执行，不得重跑或清理q6。
3. 后续每组make_binding.py run --group名称 --source-commit同上，再launch.py run --group名称 --binding对应bindings/run_名称.json --source-commit同上；先核实际现有PID/锁/组顺序。当前0_rational已启动，禁止重复队列。
ROUND_PATH=private_data/closure_mechanism_20261019/P1/rational_execution01。实际命令也在原生回执/worker记录中。全部进程额度仍取消，实际费用持续记录；未知支持成本不声称精确。相关实现及决定性证据选择性归档，原始轨迹和中间检查点本地保留。
