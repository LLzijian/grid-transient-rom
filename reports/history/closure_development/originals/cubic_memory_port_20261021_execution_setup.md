# 三次记忆端口：固定执行设置

协议cubic_memory_port_20261021_protocol.md、实际Astra原案与独立方案/实现审查先于执行。新BASE=private_data/closure_mechanism_20261019/P2/cubic_memory_port01；代码与所有source/协议/hash在首次QA前冻结，后续保持。API使用新D runtime与runner，旧模块/模型不修改、不monkeypatch旧BASE。

13700总帽：support1500(含已计planning1200)+准备QA成本1000(静态/绑定共100，唯一QA400，唯一cost500)+三train3400=10200+三eval240=720+元数据280。静态代码至此31.5658361秒；implementation_cost.json是该固定输入快照。其后每条绑定/入口准备费用仍计100并保存不可变费用补充，不覆盖已绑定快照。所有实际/失败/拒启/未核hold累计；root启动门禁必须包括native prior之外的这些实测费用，旧Q关闭费用不转入D。元数据280与support1500独立，不能事后挪账。

实际argv合同：qa400/PS370/worker340，cost500/470/440，train3400/3370/3300，evaluate240/210/180.0(float)。0<PS<Max<=3600；不以通用cap−60代替训练3300。名称小写如qa01/cost01/train_6101_d/eval_6101_d，科学variant始终D。新make_invocation只生成、不运行；command/prior/binding/launcher四文件在实际启动前归档并核远端。wrapper自身用stdlib解释器，真正worker用.venv-gpu Python；Windows原生监督记录PID和精确整数FILETIME并核GONE/源一致/两层退出。

唯一QA01包含三个新测试文件，真实CPU/CUDA，要求actualCUDA，否则失败；看退出码、全部tests与两个监督层，不只看driver JSON。没有QA02。成本单次只D12丢弃更新并占用既有两次作非空恢复见证；不创建N/Q新优化/评价，不用costtrial模型当科学source。全1024加加载/33保存/导出/监督×1.25每D<=3300、三合计<=9900，双步长16最长面板含代数/评分/压缩/39导出×25×1.25<=180；完整已花+余下最大帽容纳13700才训练。

正式顺序6101_D→6102_D→6103_D，各从原O512新Adam/同源RNG/coverage/原1024日程；不恢复/延长/缩步/减seed。三个有效1024统一冻结且归档后，按相同顺序各一次385评价。旧N/Q1024与原一次评价仅按冻结身份读，不再运行。val/test关闭、TASK4暂停，原排除/失败和真实区间保留。

代数摘要固定：稳定342事件1/2/5/8s；真实区间另列385/稳定342/失稳43。已有t0状态+真实未来n点，不重复prepend，n=1一个dt间隔。正负零时间比例按分段线性功率/精确零与实际持续时间，积分/均值梯形规则。summary=None或非有限就是未决，不能只看qualified；相关总体null、保留分母。Pextra的单位是模型H-unit/s，非机械功率MW；Hextra与其q梯度不称耗散。不追加rollout/编码或关b重评。

相关QA通过的可复用工具以后可独立工程审查/选择性PR，模型仍按原对N全精度门槛及独立审查，不改默认/不自动P4。代码/设计/失败结果每阶段选择性GitHub归档，模型/Adam/RNG/原预测本地保留。
