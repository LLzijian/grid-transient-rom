# 合并范围：Windows进程监督与Python运行环境

本次只合入三个可复用工程文件：uncapped_windows_job_20261021.py、rational_python_runtime_20261022.py及其16项运行环境测试，并附两份实际验收摘要。Windows原生Job在创建时绑定进程、保留真实PID/创建时间与进程句柄，使用KILL_ON_JOB_CLOSE清理所属子树；无数量、时间或CPU额度。直接启动venv对应的基础Python并保留原venv解释器语义，实际核对DLL、包、CUDA配置及14项float64结果。活动成员记录PID/创建时间/映像，仍按原严格规则清理并判FAILED。

固定五例实际通过：正常父子/父子孙/管道均无遗留清理，真正孤儿被判FAILED并清理，监督者121退出后孩子由内层Job关闭清理，全部Job最终0。原完整QA06为188通过，其中Job12、运行环境16；同一冻结源，native来源133项前后相同。此处采用的仅为上述工程模块通过的证据范围。

后续真实GPU训练在32步遇到诊断文件名称与共享serializer扫描冲突，完整训练入口及其原81文件集成已撤回。本PR不包括该入口或科学函数，不作模型晋升。旧失败/96更新/检查点全部保留；另登记的检查点续接正在独立验证。

原始代码与证据均可查：
- [冻结源5ffe63f](https://github.com/LLzijian/grid-transient-rom/tree/5ffe63f5f9bd71c68e417da4759fdbd183299799)
- [五例实际清单](https://github.com/LLzijian/grid-transient-rom/blob/311792005220ca5766898eb4d2d31142e8a8a594/private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/batch.json)
- [QA06原生回执](https://github.com/LLzijian/grid-transient-rom/blob/470545689dd95efd8fbb4f0eb1e2dfd5cc5790ae/private_data/closure_mechanism_20261019/P1/rational_execution01/native_qa06/receipt.json)
- [GPU32步故障归档](https://github.com/LLzijian/grid-transient-rom/tree/e429598963456c765abfcb6566f4a2caf5c5120c)

本机复现完整原验收需要该冻结源及记录的venv/DLL和已保留本地资产；不能在现有QA06或数据目录上覆写重跑。运行环境测试使用隔离合成证据，原完整验收已执行，本PR不追加验收次数。工程合并保持现有训练公式、数据划分、test权限和TASK4暂停规则。
