# 当前研究状态

任务状态：**RUNNING，阶段C已完成，正在准备阶段D**。实施用户批准的72进程小时物理闭合方程发现计划；[协议](../reports/closure_discovery_20261017_protocol.md)、[资源变更](../reports/closure_discovery_20261017_resource_amendment.md)。TASK4暂停，原冻结test关闭。

A完成实现、受控问题与成本检查；旧S-O向量场重建误差2.22e-16。部分观测例尚未恢复正确公式，证据范围见[阶段A](../reports/closure_discovery_20261017_stage_A.md)。B完成六次搜索，11个合法512次端点及三个数值失败全部保留。固定库和生成式均选同一八项结构；替代候选新增的二次表达式在已查端点可吸收到原函数族，见[阶段B](../reports/closure_discovery_20261017_stage_B.md)。

C完成四类结构各三个种子的1024次重新适配及两项完整机制删除，共14个端点；物理检查、更新次数、模型/优化器/端点哈希均通过，所有C进程已退出。main/library三个相同种子的模型字节均相同，不支持生成式方法优于固定库。main与固定结构的train排序随种子改变，仍按登记规则保留原主比较。[阶段C结论](../reports/closure_discovery_20261017_stage_C.md)和[冻结清单](../results/closure_discovery_20261017/C/frozen.json)；冻结SHA e504cb15875ce106cd834607ca28793cc539f68ccaa6d991a120db2f2128f814。

当前实际进程状态：D新增train24正在生成，实际PID和恢复命令见private_data/closure_discovery_20261017/active_job.json。repair04已通过29项合成生产检查，源码身份b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc与执行单已归档提交31f9ebe921ff745d0815a45630800d84663cd639。A阶段两个条件包含在train24内，先评价两个冻结C预测器，再对全部14个端点统一追加256次；冻结全部权重后才生成与评价24条预锁新增val。正式主比较保持main2701与fixed2701，禁止根据val重选或训练。D数据生成已启动，E尚未开始。C→D独立审查的前提及root后续核对见D/stage_entry_review.json；D当前绑定源码b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc，C及更早来源不重写。

总授权259200进程秒，A43200/B57600/C86400/D43200/E14400/reserve14400。截至2026-10-17T01:18Z的已闭账支出：A1990.0095387、B15527.1357605、C32727.9678417、D111.7186242、E41.0764439、reserve596.1004754秒；之后少量归档与准备成本尚待核算。唯一账本private_data/closure_discovery_20261017/ledger/receipts.jsonl。旧TASK2的59643.306023/60000秒单列，不转移额度。

恢复目录E:/03-nn_Transfer_Func/kiki，Python E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe。源0102 S-O SHA c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238；S-O为研究候选，F-O为开发基线，未改变部署默认模型。原train346个合格事件、35个失稳样本，100ms因果边界和真实区间保持。

GitHub归档分支codex/closure-discovery-20261017；关键设计、代码、正负结果均归档，原轨迹、缓存与重复中间检查点留本地，重要模型记录路径和SHA。无关修改、本地HEAD a951013及用户index保持。恢复入口见[执行说明](closure_discovery_runbook.md)。
