# 固定边项完整接口与费用核验：结论

本轮完成两结构的CUDA数学和恢复小样本检查，以及完整八项结构18次费用试算。完整核验在带完整来源身份的检查点恢复处失败，删除组试算、两结构全366train评价及费用表未完成，**P0费用准入未决，不启动P2**。失败不评价删除机制的科学作用。

两次QA按预登记容差通过，检查完整八项/删除整项后的七项、场/能量梯度、原目标全分量及参数梯度、None模式、Adam、导出、恢复与下一更新。QA实测46.1312582秒计800支持内；唯一修正用于Windows自有进程树终止，QA和修正次数已用尽。独立edge_result_challenge审查接受的是以全部实际检查及完整测量为条件的工程范围，不能替代未完成费用门槛。

## 决定性执行证据

实际计算PID67828、监督82708及launcher52620/60876均已退出。单批EXEC实际**121.16303799999878秒**（监督包含子墙钟只记一次）。共36计划更新中完成18，全部为full8的1/2/5秒×三个预先冻结train面板×两更新；试算权重不作为候选。delete7尚未开始。该原始计时全部保留，包括较慢值；没有重测择优。日志记录GRU参数未连续存放的警告，未修改运行路径以改善本轮成绩。

带原Adam状态及1024步日程元数据的full8_discarded_probe/step000002已保存；restore_checkpoint在比较来源身份时抛出ValueError。磁盘checkpoint身份与frozen_identity JSON相等为True，这不等于运行期对象比较通过。当前代码结构使用元组记录部分参数身份，JSON读回列表可能解释该差异；该原因属于待后续有限方案核实的实现问题，未放宽或绕过原检查。原始错误栈见紧凑证据。

源driver SHA256 5a15cf58e5c3e56ce9e38a01e59ca7e06ea9e49eeb0256cca64f13ca42433fcb，tests SHA256 52af0cd754cb0dbc2d2dedddb88ba4236d5d296ca16833709aa0143f8ae40cf0，执行前后全部40份来源哈希一致；冻结身份SHA256 8c3302dc0e810c1c42df2f6a54f3f430ea14ae870a2cfea7c850453e8a9ff263。原S-O c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238、366train身份42228e81d9f3d15c94350b14ae138a39285f19448db4c2dbc5fd376c0e513685均由原接口核对。公开main包只提供八项terms，未采用已训练权重。

## 关闭与费用

本轮3600分配中的EXEC 121.16303799999878秒及800支持保守扣留；支持不是实际消耗，QA46.1312582不再另加。未消费EXEC 2678.83696200000122秒明确撤销。P0保守剩11753.4159736000秒，父剩231735.5063507000；P1仍3982.0903771、P2上限86400且未启动。原900未核对支持继续保留，旧账本及关闭额度不转移。

本轮不追加第二批、不继续修复或QA，不将部分较便宜的测量当作完整准入。未接受的运行入口不合main，默认模型不变；源、有效小样本检查与失败记录保留归档分支，未发生需要revert的main集成。下一科学轮已实际调用gpt-6-astra /root/astra6_after_edge_gate_failure只读规划，须前瞻登记后执行，不把下一动作视为在跑作业。原test关闭、TASK4暂停，当前无训练或测量作业。

## 复现与访问

实际命令：`.venv-gpu/Scripts/python.exe experiments/edge_pair_cost_gate_20261019.py measure --output private_data/closure_mechanism_20261019/P0/edge_pair_gate01 --max-seconds 2800`。既有目录不能覆盖或继续测量；命令保留复现信息，不授予重跑许可。

必要代码、协议、紧凑QA及结果位于codex/p0-edge-pair-gate-20261019。results/edge_pair_gate_20261019/closure_summary.json包含全部18原始计时、错误、回执和重要身份。完整frozen_identity、schedule、checkpoint/Adam、日志及真实366train保留本地private_data/closure_mechanism_20261019/P0/edge_pair_gate01；复现实验需要本地原数据和S-O来源，不全部包含在GitHub中。
