# Windows启动重构的实际观察

独立挑战指出记录接口闭环问题后，按既定一次机械修正执行A2。来源1beccf4f2d3313e26ecbf3f524a4b6d7f6939295；normal再次COMPLETED/无遗留清理/Job0，orphan再次FAILED/实际清理/Job0，源码前后同。两类子进程身份均绑定并实时核GONE。A2 normal wall5.1310882、CPU user2.1875/kernel1.234375；orphan wall1.0220365、CPU user0.0625/kernel0.109375。公共参考未重复、B未运行、无第三次A观察。A1选择保留为runtime_selection_direct_base_venv1.json，最终选择A2的runtime_selection.json SHA256 **38f8135cc2f087e27c9c02b04c4ead2496950d5e71d2aeaf274e2e0b977f1bcc**，15文件绑定。下文A1为原始首次观察记录，不再是最终选择。完整QA04尚未执行；最终代码冻结后才执行。

首个方案direct_base_venv通过预注册的机制观察，选定用于唯一最终QA04。正常父子程序COMPLETED，真正遗留应用子进程的负例FAILED并被清理；两例均Job0、真实创建身份已记录、源码前后一致。尚未运行QA04或正式训练，这一结果只支持进入完整工程验收。

来源为已核远端e5ece28c92dce4ccab90953a7e3e554ef2662d2f。公共参考与候选均使用同一probe_worker.py，Python3.12.14、实际Python DLL及哈希、包路径、Torch/CUDA、确定性设置和14个float64 CUDA平方值逐字段一致。没有赋值sys.executable/prefix、没有读取科学数据。原venv参考此次正常程序实际COMPLETED，保留该事实；QA01/02/03历史失败均未改判。

| 运行 | 原生结果 | wall秒 | Job CPU user秒 | Job CPU kernel秒 |
|---|---|---:|---:|---:|
| 公共参考normal | COMPLETED，无遗留清理 | 9.966050899998663 | 2.421875 | 1.578125 |
| A1 normal | COMPLETED，无遗留清理 | 5.170692499999859 | 2.203125 | 1.343750 |
| A1 orphan | FAILED，实际清理已握手子进程 | 0.9962400999975216 | 0.062500 | 0.109375 |

这三次是不同控制程序的费用记录，不构成训练加速比较。A1 worker分别为44460/FILETIME134371255531771251与45724/FILETIME134371255583758136。候选负例孩子的完整identity及GONE检查保存在原始证据和选定文件校验中。

选定文件private_data/closure_mechanism_20261019/P1/rational_execution01/reconstruction20261022/runtime_selection.json，SHA256 68a9d453c85ac2ed7e6259313bbce7e1435e4f1131573db21b73241680b43b8f。它绑定14个输入/证据文件；构造命令每次重读哈希、两原生回执、环境与张量原值以及实际孤儿身份。所有科学模型与默认采用保持原状。A未用机械修复，B未启动，完整QA04次数仍0。

复现先恢复仓库及本地同一venv依赖，将归档reconstruction_reproduction中的程序放回上述目录；使用基础Python运行observe.py --scheme reference --attempt 1 --source-commit e5ece28c92dce4ccab90953a7e3e554ef2662d2f，再运行同程序--scheme direct_base_venv --attempt 1，之后select_observed.py direct_base_venv 1。原执行目录专属且不覆盖；复现须独立副本。观测原始JSON和必要日志选择性归档，原始路径与哈希完整保留。最终验收仍须独立挑战和冻结源，不以本报告替代。
