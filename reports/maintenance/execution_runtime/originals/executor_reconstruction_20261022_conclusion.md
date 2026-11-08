# 启动重构：机制通过，完整验收因临时路径报错

QA04实际为**73项通过、26项测试设置报错**，完整验收FAILED，原12组GPU训练未启动。A2启动机制的正常退出、真正孤儿清理、环境一致性及源码绑定证据成立；最终套件中的26项准入检查未执行到断言，不能将已有通过项当成完整放行。

唯一QA04使用已归档源6eb63542ee4da28be4227d653d32f3d48bd8367f，实际worker47152/FILETIME134371262453630886。pytest78.81秒；原生wall81.6594533秒，Job CPU user19.375/kernel31.46875秒，launcher/worker exit2，Job19个累计进程、最终0，cleanup=true且无遗留清理；100个输入源码/绑定文件前后一致，锁已清。原8项数学测试及Windows Job、运行环境套件通过；全部26个错误来自同一个reconstruction_binding设置入口。原始完整stdout、binding、native回执与摘要均保留。

实际错误发生在创建286字符的原子临时文件名，父目录存在且长226字符；漏检了Windows临时路径长度。不是GPU、训练或物理模型失败。已准备的具体修正仅将QA专用pytest临时根目录从输出目录内pytest_tmp缩短到仓库private_data/q4；对应文件名降至222字符。没有改变量、损失、数据、数值或采用门槛。该补丁只做AST检查，尚未复验；QA04原始源码另保留在本地qa04/failed_source及原提交，失败结果不改写。

本轮人类批准的问题明确限定“一次最终完整验收”，该次已执行；未运行QA05，未启动训练，没有合并main或改变默认采用。协议的B仅在A机制不合格时启用；A2已合格，B不能替代失败的完整QA，亦不以改名获得额外次数。进程额度仍全部取消，当前阻塞与费用无关。继续完整验收需要新增一次明确授权，之后仍先经真实Astra6有限规划与来源核验，通过即自动执行原12模型；无需再次询问是否训练。

科学状态：本轮为实现/验收问题，正式比较未执行，不能判定模型族优劣。保留全部A1/A2、公共参考、QA01/02/03/04失败和各源检查点。原项目val/test关闭、TASK4暂停、40状态/100ms一次因果初始化/真实区间和失稳样本标准保持。工程改动仅在归档分支留存，未取得完整集成资格。

复现命令见results/rational_execution_20261021/reproduction/launch.py与make_binding.py；原执行以基础Python运行launch.py qa --qa-attempt 4 --binding private_data/closure_mechanism_20261019/P1/rational_execution01/bindings/qa04.json --source-commit 6eb63542ee4da28be4227d653d32f3d48bd8367f。原目录专属不得重跑覆盖；复现需独立副本和相同本地venv。取消额度后本轮观察原生wall22.2861082秒与QA04 wall81.6594533秒分别入账，CPU另列，支持成本仅报告已知部分。
