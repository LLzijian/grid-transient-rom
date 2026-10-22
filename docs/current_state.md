# 当前研究状态

任务RUNNING，cwd E:/03-nn_Transfer_Func/kiki。人类2026-10-22已明确授权“允许重构并继续训练”：两个预定启动方案、一次最终完整QA04，通过后原12组GPU训练。实际gpt-6-astra原始方案/root采纳及有限协议已归档。当前A1机制观察合格，独立只读挑战进行中；完整QA04未启动、正式训练0、无科学PID/队列，正式合成val未生成。

**决定性证据**：来源e5ece28c92dce4ccab90953a7e3e554ef2662d2f。direct_base_venv正常程序COMPLETED、真正孤儿负例FAILED并实际清理，均Job0/精确PID和FILETIME/源一致；Python DLL、包路径、Torch/CUDA、确定性及14项CUDA值与公共参考相等。A1未修复，B未运行。选定文件SHA68a9d453c85ac2ed7e6259313bbce7e1435e4f1131573db21b73241680b43b8f，14个文件绑定。[观察结论](../reports/executor_reconstruction_20261022_observations.md)、[有限协议](../reports/executor_reconstruction_20261022_protocol.md)。该结果只支持进入完整工程验收，不是模型改进。

**下一可执行动作**：完成非实现者只读挑战，冻结并归档最终源，生成qa04专属binding并运行唯一完整QA04（工程三套+原8项数学）。仅内层全通过且外层COMPLETED/无遗留清理/Job0/源同/实际身份通过，才立即执行0_rational、0_tanh、1_rational、1_tanh各三seed512更新。保留0/32/128/512检查点和32处真实恢复；全部12端点冻结后一次合成val。无QA05，无384重测，无旧18×512重跑。执行入口private_data/closure_mechanism_20261019/P1/rational_execution01/{make_binding.py,launch.py}；新观察恢复目录其下reconstruction20261022。

**科学边界**：原项目val/test关闭，TASK4暂停；40状态、100ms一次因果初始化、全部真实区间与失稳样本、原数值和模型晋升标准保持。原12模型为两形式×6301/6302/6303×0/1%噪声，共6144更新。没有正式模型结果可晋升，默认模型不变，父L1/L2目标未完成。旧P3条件与24valguard限制保持。

**历史保持**：QA01未决、QA02失败、QA03 45通过3失败均不改。QA03来源95ee7cfe3d0c9dbdc065a306fcdc1d60fd0078d3，native wall55.9809651、Job0/源同/锁清；归档e82a5a3a883b6d7aed5d24eb0e54fdacb68c23fe。新重构明确授权JSON归档c42782d323593b21d69e241ae84306699bf9cbfd，协议24ece7ec822c1d47b6d7e08390e3ae1d72e754fe。

**费用与授权**：全部进程数量/时长/阶段/父小时额度取消，实际费用照记；旧259200、分类余额、未知hold及REJECT均仅历史。旧18×512为6085.6472446秒。重构公共参考+A1两例原生wall16.1329835秒、Job CPU user4.6875/kernel3.03125另列；旧取消额度后QA/诊断原生wall124.9277505保持历史小计。未完整测量支持成本不冒充精确总实耗。详见results/closure_mechanism_20261019/cost_snapshot.json。

**Git与恢复**：当前新执行器未合main；归档分支codex/edge-memory-discrimination-20261021保留全部来源/失败。政策PR29 main ad1b48a3b701cfaf951d0e96067d5fb537e74c90已核。保护local HEADa95101337855ed3fc1f4abfd710982443d5564cb与indexSHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93，不reset/强推/删证据。工程资格通过后的精确范围PR合并另核远端；工程合并不等于科研采用。heartbeat ACTIVE，仅可信变化/完成/关键失败或需用户行动通知。