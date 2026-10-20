# 当前研究状态

任务状态：**RUNNING**，目录E:/03-nn_Transfer_Func/kiki。当前执行[空间极值监督协议](../reports/spatial_extrema_20261020_protocol.md)：actual gpt-6-astra代理/root/astra6_after_temporal_execution[原案](../reports/astra6_spatial_extrema_20261020.md)及独立方案/实现挑战已由root采纳。固定七项40维和1/2/5s，比较原目标O与物理角差范围/最大COI相对转速联合监督S，三seed。只读复用未评价的旧6101_O，新训练顺序6101_S→6102_O→6102_S→6103_O→6103_S。原控制费用不重复计，六角色全部有效并冻结后才统一各评价一次。

**实际作业**：五次新训练全部完成512/385覆盖，真实训练累计6480.491851099941秒，五次两层0/GONE/源一致，RNG/日程/目标/导出通过，无重试。六角色冻结manifest SHAb8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4，启动评价前已归档6341e186767234dee8a1a7daefa9db592e86df30。统一评价已完成6/6，每次385完整事件、全部保留；最近完成6103_S、远端538ad78680fec7300239283efcc211b1b868559e。当前全部六次评价完成，无科学进程；root下一执行JSON-only统一提取与预登记门槛判断。执行分支codex/spatial-extrema-runs-20261020，root/heartbeat不重复队列；不追加训练、重试或768。

**准入证据**：QA01 24通过/6失败均为Double/Long计数类型错误，唯一一行修复后QA02实际30项CPU/CUDA全部通过（其中runtime11参数化用例），测试字节未改。prepare12.4067707秒核共同385身份、显式十机平衡角、旧O512原日程/来源；cost53.7889218秒核六来源初始等价、O/S各12次完整更新/真实5s非空Adam恢复。全部退出/cleanup GONE/源一致。含25%余量的五训练8645.4582503/六评价643.84605秒已准入。frozen_training_batch SHA80bfe2deca98b0a5d1b25936138d9e63d90e0776c3c8eb9c4bb27d7e2aedd39e，正式源码77ea266d00a544d0b029b7a8723c8bef6cbe9fa9。

**费用**：父上限259200秒，当前累计102963.2981572/剩156236.7018428（含保守支持hold，非全部实测）。本轮P2登记24000：QA4200、五训练12000、六评价/提取4000、支持3800含已计1200规划。已发生工程108.0690704、五训练6480.4918511、评价及冻结/提取准备647.8833152秒；轮内未结算12963.5557633，当前评价在途cap0。父未分配143273.1460795、P2未分配20492.0524717；其他类别余额P0 9037.2289973/P1 3982.0903771/P3 75986.9220123/P4 21600/repair12174.8522211不互借，旧900hold保留。[累计账本](../results/closure_mechanism_20261019/cost_snapshot.json)。

**工程与提取**：独立接受的空间runtime/纯合成测试/说明已通过PR17合main并核实3e97af12092545ad727e216e4afb33adb0d405f7，三个blob精确一致，八项已有公开计算依赖身份不变。PR16时间目标工具仍保留；两者均不采用模型、不称预测收益或加速。新JSON-only提取器5项合成测试通过，SHAe4f5494e2cd6feb14e53cef1ccbfe8f70eee1bbdfae9961fc2bfb85ffa4ae9ca，未读新成绩；正式提取须六评价完成并传冻结manifest SHA，原指标与空间解释均传播相关数值未决，事件rows留本地。

**历史与边界**：时间目标O/F/T因F cap3850被公共PS硬3600在worker创建前拒绝而关闭；仅O6101完成512，旧轮共5226.9986015秒，未用36773.0013985取消回P2另行登记；不改旧门禁、不复活F/T/旧768。quadratic C相对B失稳点误差下降12.66%–16.01%但稳定5s风险全升，真实漏检B28/28/27、C29/26/28，C不采用；A6101 expansion_0028的8s资格未决保留。B/C的39D Hurwitz及线性witness不等于非线性稳定域；P4尚未满足同一候选预测资格，保持不开启。

**数据/恢复**：共同385=原366+新19（342稳定43失稳），新19已用于训练，原27排除/新5失败保留；按冻结源身份分栏。100ms一次初始化、真实时间范围/单位/数值标准保持。val/test关闭、TASK4暂停、P1真实搜索关闭、旧18组512耗6085.6472446不重跑、旧24 val不重试/搬模型绕guard。新BASE=private_data/closure_mechanism_20261019/P2/spatial_extrema01；importO仍在temporal_objective01/models/6101_O/checkpoints/step000512。模型/Adam/RNG及全部不利记录保留，原始轨迹本地、关键路径/hash归档。root分支codex/spatial-extrema-20261020。local HEAD a95101337855ed3fc1f4abfd710982443d5564cb/index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93及用户工作保持。heartbeat ACTIVE，健康无变化保持安静。
