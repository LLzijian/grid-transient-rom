# 执行设置与恢复

本轮仅新增 `experiments/rational_control_20261021.py`、对应测试及P0/P1的 `rational_control01` 私有执行助手；历史源码、原搜索语法、默认模型和用户Git暂存区不改。两个工人分别拥有核心/测试和执行助手，不相互编辑。实现采用统一空项base、同形零端口系数和有理/tanh函数标识，原GRU及六历史初始化器共享实现。

全部数值仅由GPU环境 `.venv-gpu/Scripts/python.exe` 在原生监督内执行。标准库启动器使用 `C:/Users/LLLzijian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`；cwd为 `E:/03-nn_Transfer_Func/kiki`。每个阶段先运行对应目录 `make_invocation.py` 生成一次绑定命令，后经 `observe_launch.ps1` 调用原生两层监督。P0 phases为qa/cost；P1 phases为run/evaluate，run group固定0_rational、0_tanh、1_rational、1_tanh顺序。source-commit为首次QA前归档的完整40位源码提交，QA后不改绑定科学源码。

P0支持配置为450+QA250+cost450+关闭100=1250；P1为四run各2750+eval650+冻结50+支持300=12000。worker自限分别qa220、cost420、run2700、eval600；PS超时235/435/2730/630；native帽250/450/2750/650。原生单进程不超过3600。任何旧回执未核清、锁仍在、来源变化或失败均不重复排队。每个进程按实际子树费用只计一次；PID与整数FILETIME核真实创建身份，退出/GONE/锁清必须通过。

P0绑定前一份实际合格长时QA的96个历史科学文件及本轮AST依赖、全部新数值/测试/原生助手、协议/采纳/独立计划挑战/本设置。实际QA通过并核前后源码后，cost须完全相同科学身份。原生外部单调起止及生成命令实际wall保存；完整cost后root计五个科学进程启动，保留主体未分类开销。仅在费用与分进程25%限制全部通过后生成P1 admission.json；该文件含实际numerical_source_commit、immutable_science_files、PASS及P1_complete_cap_seconds12000。

四个run selection原样合并为统一12端点selection.json，核实际512、trace、非空Adam、source、数据配对和所有SHA后evaluate才可生成合成val。任何组未全冻结即停止剩余队列；不以其他组结果代替完整比较。最终trace、逐事件评价、重要原始数据与真实32/512checkpoint保留本地；紧凑证据、路径SHA及复现命令选择性GitHub归档。

首次QA前独立实现挑战要求补齐Adam状态、两形式NumPy自主推演和cost实际trace读取；实现冻结前完成。cost trace真实32步，加cost_only填充覆盖完整序列化上界，两次load核验均读盘/hash。P1固定元数据50秒覆盖root统一冻结及helpers对科学端点、原始数组、trace的额外重复SHA/来源核验；原生费用按完整子树计一次，保守元数据不称精确实耗，不挪给训练。

当前本文件是首次QA前设置，不声称QA、cost或训练已经运行。各次真实PID/费用和门禁结果另由实际回执给出。尚无本轮模型采用决定。
