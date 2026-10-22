# 检查点名称修复与固定12模型续接

实际指定gpt-6-astra(/root/astra6_qa05_plan_20261022)原始方案另存astra6_checkpoint_continuation_20261022_raw.txt。root采用有限迁移与原比较续接路线；依据AGENTS连续自主授权及人类原训练许可，不重复询问，不追加QA07/不重跑五例或QA06。旧三UNRESOLVED组闭合、96更新、检查点原件与费用不改。

工程只新增专用模块/入口/必要测试，旧core/test/Job/helper/probe和54数值依赖字节保持不变，避免把旧运行资格的源hash换掉。新模块显式调用原study数据/make/advance/verify/evaluate与原共享checkpoint/restore；不在内存替换旧核心函数。诊断统一diagnostics_00032.json等非checkpoint_*名称；shared pending/更后项/hash/identity/RNG检查不变。

先保存三行原模型来源清单（全部原目录内容与原文件sha/bytes/原identity/源），固定6301/6302/6303、仅32来源。原文件未知/pending/更后检查点/sha/计数/诊断不一致即停，不选其他点。加载前复制已全部核对原0/32及latest到独立只读恢复副本，侧文件改新名字置扫描之外，原件不动。实际shared restore必须用原identity/原training.pt，核实际32、Adam、NumPy/CPU/CUDA RNG与诊断loss。新science输出初次保存32用新identity并显式origin原文件引用，不能称全512都由新源产生。迁移三条一次整体资格，不选择性换种子/重训。

固定一次相关非GPU测试集合，用真实共享serializer检查新文件名、pending/更后/篡改/缺诊断拒绝、来源迁移、旧complete不可当中断、新12端点屏障。只新模块相关测试，非全套QA07；发现失败保存并按本阶段有限修复0停止本批，不修后重复。实现前静态修正不限于未执行代码，不是运行重试。

固定一次CUDA资格：旧6301无噪声train，rational与tanh各连续0→33、0→32实际保存/重建/shared restore→33，共4路径132更新。记录实际运行配置；不通过改变既有科学函数/学习率/线程以“修”轨迹差异。目标状态逐值一致（Tensor/Adam/NumPy/CPU/CUDA RNG/完整loss与梯度），无结果后容差扩大。rational32还对照旧6301真实32状态。三份旧32各加载一次且不额外更新；旧rational6301加载可同时用于对照。四路径+三迁移全部通过才正式6048。错误/差异即保存关闭，无自动重跑或全新12替代，原val/test关闭。

正式顺序0_rational三迁移各480，0_tanh/1_rational/1_tanh各三seed各512，新增6048，旧96单列后总正式6144；所有模型512才一次原合成val。新模型从0保存0/32/128/512并32真恢复，迁移模型首保存32再128/512、不额外恢复32；所有loss/梯度轨迹完整512。最多一次非择优外部中断恢复沿原标准， completed marker不能进恢复。固定无指标驱动调参/分支。

新目录private_data/closure_mechanism_20261019/P1/rational_checkpoint02，原Job/真实PID/FILETIME/源前后/锁/实际wall与CPU照记，无进程额度。selected_python_command继续核旧不变的资格原件，新native额外绑定新增模块/协议/本次实际资格与三行来源，不把旧QA06伪写成新源通过。旧模型文件/来源不改，test/原val闭合，TASK4暂停、40状态/100ms一次因果接口/真实区间/失稳样本/数据配额/原科学数值门槛不改。

正式采用前独立只读挑战。源、相关单元结果、132更新资格、迁移3行资格与所有结果归档GitHub；不合入当前已知故障训练入口。新工程相关真实验证通过可单独精确PR，模型采用仍依科学门槛。任一失败后关闭当次并选下一可辩护阶段，不把取消进程额度当无限重试。
