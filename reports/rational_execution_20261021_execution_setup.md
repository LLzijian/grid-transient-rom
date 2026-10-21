# 本轮执行设置

数值核心保持原8项合格QA字节，54项实际数学AST闭包据原native QA逐项SHA256及已核Git blob身份验证。AGENTS和工程流程政策更新另行绑定，不把旧118整map说成未变。

标准库环境 `C:/Users/LLLzijian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` 运行 `private_data/closure_mechanism_20261019/P1/rational_execution01/make_binding.py qa|run|evaluate --source-commit FULL`；run另带固定group。生成绑定而不执行。随后标准库同目录 `launch.py` 携对应 `--binding`、`--source-commit` 和group启动Job。真实worker是 `.venv-gpu/Scripts/python.exe -m experiments.rational_execution_20261021`。cwd始终项目根，不经其他shell删除或移动文件。

新CLI没有max-seconds和quota；所有数学函数deadline=None。Windows Job创建时以JOB_LIST原子关联挂起worker，限制标志只有关闭句柄时清理作业树，没有时间/CPU/数量帽；短间隔等待循环仅使取消和真实worker身份核对可处理，没有累计截止时间。记录真实worker/监督器PID与整数FILETIME、所有源前后哈希、退出、Job活动进程数0、墙钟及累计子树user/kernel CPU时间；墙钟与CPU不相加。实际QA包含新薄入口与新Job两份tests，旧数学8项不重复。

QA正常/异常/监督器恢复线程前后硬退出仅使用短小工程子进程，不触及正式seed数据。测试握手观察期限用于判定测试失败，不是正式作业时限。独占锁拒绝重复启动；真正Job清空回执和精确创建身份均核实后才可显式清残留锁，不按年龄猜测。

恢复最多一次且只针对未完成的非择优中断。绑定实际旧Job回执，核source/group/workerPID及FILETIME/Job0/源前后和真实费用；不信布尔批准就放行。旧worker证明先复制为不可变副本，不能将新worker或progress将改写的路径放入来源哈希。checkpoint/Adam/RNG与完整梯度侧文件须同一保存点，缺失或部分发布则未决；未知重算步数报告区间，不捏造精确计数。监督器直接硬退出没有回执时不声称可恢复。

四个group各3seed、每模型512，全部12端点冻结才一次评价。root用标准库合并四组selection原始rows，保留12真实512端点的source/model/training/train_raw/trace哈希；evaluate继续调用旧完整barrier，前后复核。某组科学未决时保留数据并可执行其余独立组合，不通过删组提前打开val。

原项目val/test关闭、TASK4暂停。输出新P1/rational_execution01，原历史P0/P1资产不覆盖；旧18×512和384cost不重跑。取消额度授权并不意味着训练直到通过；本轮科学规模和修复规则见protocol。当前文件在首次QA前登记，不声称QA或正式训练已经完成。
