# 唯一实现修复的独立复查

审查者 `/root/fom_history_interface`，非实现者。实际只读4.1458501秒，本轮累计22.7445122秒；没有文件编辑、数值运行、pytest或新cost。

原始实质返回：

唯一修复小范围只读复查通过，可执行QA02；同源QA02与native监督均实际通过后，可直接启动固定科学比较。五份最终SHA均与所报完全一致。新entry的atomic仅对PermissionError winerror5/32/33有限6次replace（5个固定短退避），其他错误立即失败，永久失败保留旧目标和唯一tmp；旧数学atomic/field未改。新增QA含短暂恢复、永久保留、其他错误不重试。

监督现在分别持有launcher与真实worker句柄，实时capture核typed PID/FILETIME、GetProcessTimes、IsProcessInJob、live wait状态、source/phase/group/qa_attempt；记录物理image，直至实际worker退出，不靠receipt字符串或强行PID相等。settle_job对Job成员句柄核live：仅真实live成员触发TerminateJob，已signaled但accounting滞后只等待归零。Job仍无时间/CPU/count限额。新增实际venv/redirector收到身份后才release的测试，覆盖真实worker捕获。

launcher对旧qa01只允许确切历史source0a395.../UNRESOLVED且原source前后/cleanup/Job0/hash保持，QA02必须独立输出并绑定repair登记；正式阶段只认native_qa02 COMPLETED+worker PASS且实际新required-current字节同源，没有把历史qa01改作PASS或开放qa03。原17项扩展后的新增两测试仍统一运行，不数学重测。

这是静态允许QA02，不声称实际已PASS；若QA02仍失败，按登记规则隔离，不能再修到通过。本次native只读4.1458501秒（2.2226650+1.9231851），本轮自协议以来累计22.7445122秒，纯实耗记录；无编辑/数值/新cost。

root采纳：修复限新执行/监督与QA接入，原科学内容保持。源码统一冻结归档后执行唯一qa02；通过即开始四组正式训练，不追加费用准入。
