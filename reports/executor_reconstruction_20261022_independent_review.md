# 启动重构独立挑战与root处理

非实现者/root/reconstruction_independent_review_20261022实际执行只读检查，未编辑、未运行pytest、probe或科学计算。审查范围为本轮Job执行器、命令构造、科学入口与绑定、三套工程测试、启动脚本和A1原始证据；不把这次审查作为训练放行。

原始结论：当前版本有执行闭环阻塞，需在A2观察和唯一QA04冻结前处理；A1原始结果支持方案A的环境一致性与严格孤儿清理。

1. Job实际worker只有PID/executable/physical_image/times，而训练准入和恢复另要求logical_executable/runtime_scheme/runtime_selection，实际运行会拒绝。
2. 选择程序固定独占创建既有文件，A2原命令会因已有A1选择失败，应保留A1副本后发布A2，不能覆盖历史。
3. QA04准入应明确要求exit_code/launcher_exit_code/worker_exit_code均为整数0，而不只依赖COMPLETED。
4. 建议将正常child.json也绑定parity，核对两类孩子的PID/FILETIME/逻辑解释器/GONE。

审查实际核A1选定14文件的SHA一致、公共参考/候选metadata逐字节同、正常COMPLETED/无清理/Job0、孤儿FAILED/清理/Job0且孩子PID已不存在。未发现改变科学定义的路径。取消和崩溃清理仍由唯一完整QA04验收。

root采纳全部发现：Job从实际句柄生成物理身份及创建时间，逻辑与方案核对已发布信息；选择引用规范路径和SHA吻合，保留worker引用形式以使准入/恢复同值。准入增加三个typed0退出码反例。A2选择先验证，再显式将A1选择重命名留存后独占发布新选择；观察证据不覆盖。两个孩子身份均进入绑定并实时核验。

这是预注册每方案一次机械修正后的唯一复核，修正仅涉及同方案记录与调用接口，不更换机制、不调整等待时间或判据。先归档修正来源，再执行A2固定normal/orphan组合；公共参考不重复。A1和旧历史保持。若A2合格，冻结观察源码不再修改，最终工程源归档后执行唯一QA04；它合格才自动继续原12组GPU训练。QA04尚未执行，无科学放行。
