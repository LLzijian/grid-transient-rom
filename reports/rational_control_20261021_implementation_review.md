# 首次QA前独立实现挑战

审查者 `/root/fom_history_interface`，未参与实现；本次工具8.0536998秒，计已登记开跑前review100（此前计划挑战3.8866818同属该额度），不另收费。

原始实质返回：

已读core/tests及P0/P1助手，尚未见teacher/噪声/参数配对的确定错误。两个需在唯一QA前处理的具体合同缺口：verify()的Adam校验只检查非空opt.state和已有step；若state非空但所有项无step则any(empty)=False会通过，也未要求param_groups/每组参数有对应真实状态；至少恢复旧严格模式的nonempty step列表、param_groups和所有已训练参数步数/必要Adam状态检查，并加缺step/空groups负例。另协议写独立导出“推演”，当前QA只在一个状态比较numpy_field与field，没对两形式做独立NumPy RK4自主轨迹核对；建议用固定非零coef、小批量/固定50步的已有容差核对，避免仅字段一致就称独立推演通过。

费用还有一处明确的新trace读取漏项：正式verify(cost=False)会读取并SHA核每模型trace.json（512loss+梯度摘要），且evaluate前后各一次；cost verify(cost=True)直接用内存cost_trace，完全不读trace文件。JSON面板只计写盘。请在唯一cost的load里实际读/哈希带真实32条loss、另加固定cost_only填充覆盖512序列化包络的trace文件，保持真实step32，勿伪造512。P1生成/execute selection与source_before/after还对端点、train_raw、trace重复SHA；这些新增资产检查也须明确计入实测load保守次数或50秒元数据项，不能认为P0仅源码的startup自动测到了科学资产I/O。其余全384/48/12/24/12/36/256次数闭合，五进程帽均≤3600。

本次只读实现挑战结束：条件接受，首次QA前须收束已反馈的3项：严格Adam完整性（不能any(empty)通过，含groups/step/必要状态）、两形式独立NumPy RK4推演（不是仅field点值）、cost真实trace读取/SHA及科学资产来源检查费用覆盖。末次rg已见你新增validate_optimizer和NumPy轨迹段落，但未完整重读这些新补丁；最终采纳/冻结由root负责。没有发现其他确定阻断项：空项base+同形wrapper系数与自有RK4正确，optimizer闭合1e-3只收一次；clean/noisy/scales/val未来分离及批次/参数配对成立，12端点barrier、四run顺序和不完整组阻后续均存在。P0/P1 native帽、PID/FILETIME/GONE和唯一QA/cost来源链静态可对应协议；实际数值是否通过仍待唯一QA。工具native wall合计8.0536998秒（1.8560757+2.0859321+2.0879093+2.0237827），计同一开跑前review100；未写文件/改代码/运行数值/pytest/Git/递归。

root采纳三项，在首次QA前补齐，冻结后不修改。缺梯度的合法骨架参数可以无Adam状态，已有状态必须完整且步数准确；不把合法未训练参数误判为失败。cost必须实际两次读/hash填充trace并比对真实32loss。另固定P1元数据50秒覆盖root统一冻结与helpers对端点/原始数组/trace的额外重复SHA和来源检查；它不是可挪训练的余额。原生子树实际费用仍只算一次；该50保守预占不冒充精确实耗，不声称P0源码startup已测到新增科学资产。
