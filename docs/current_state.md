# 当前研究状态

最新任务：**RUNNING：2026-10-19用户已授权按改进方案推进**。新上限72进程小时，P0/P1先合计12小时，旧账本关闭。见[执行协议](../reports/closure_mechanism_20261019_protocol.md)与[原方案](../reports/closure_improvement_plan_20261019.md)。P0已启动完整GPU成本探针；实际进程与测时保存于private_data/closure_mechanism_20261019/P0/measure01/measurement.json，探针模型不采用。下一步：核对366条train与费用后执行因果恢复，满足门槛才进入真实搜索。原test关闭，TASK4暂停。

任务状态：**COMPLETED：本轮GPU训练与工程效率更新已交付**。所有验证进程均退出；本轮未启动真实数据研究训练，没有后台任务。下一次40维闭合模型训练使用默认CUDA入口；原test关闭，TASK4保持暂停。

[本轮工程结论](../reports/gpu_training_20261017.md)、[GPU运行说明](gpu_training.md)、[工程范围](../reports/gpu_training_20261017_protocol.md)。新增experiments/closure_training_runtime.py和train_closure.py；独立环境E:/03-nn_Transfer_Func/kiki/.venv-gpu/Scripts/python.exe，已验证RTX 4060 Ti、PyTorch2.13.0+cu130、CUDA13.0。旧CPU环境不变。

35项CPU/CUDA检查全部通过：损失与全部可训练梯度、一般表达式、变长真实区间、公开方程包的场/Jacobian、导出、Adam及CPU/CUDA/NumPy随机状态恢复和中断回退。固定0.2 s合成前向反向、12/96事件的GPU同设备加速分别为1.88/1.86倍，CPU同设备为1.84/1.72倍；优化CPU在这两个批量下仍更快，不声称GPU比CPU加速。float64、0.01 s RK4、100 ms一次初始化、原采样及discover_closure.loss定义保持。

下一可执行入口：`.venv-gpu/Scripts/python.exe experiments/train_closure.py doctor`。研究训练按新的来源、更新次数和额度登记，使用`train`子命令及新输出目录；恢复加`--resume`，更换设备需明确记录。合成速度复核使用`benchmark`子命令，不打开数据。检查点stepNNNNNN/latest.json；GPU源模型仍从登记的symbolic_closure_v1导入，不能覆盖旧冻结模型。

工程费用快照：验证/支持506.080/3600进程秒，软件安装1322.489秒单列；详见[本轮费用](../results/gpu_training_20261017/cost_summary.json)，本地逐命令审计private_data/gpu_training_20261017/native_audit.json。未从旧72小时计划转移余额，快照后的归档核验按实际回执继续计费。

上一有限A—E研究的[最终报告](../reports/closure_discovery_20261017_final.md)保持：main/library三个同种子模型相同；部分观测结构恢复及生成式优势未成立。锁定新增val24中23可评价，完整确认未决；主候选在3条可评价失稳样本中漏检1、固定对照漏检0。主候选存在局部正实部模态，未获非线性稳定域证书。全部仍为研究候选，S-O为研究候选、F-O为开发基线，部署默认模型未变。

原研究源码身份b8d1f961bd5ae7700b8c656b28357b0ad537a77bd838fda5530c4c0c0f4694bc，30个绑定文件哈希保持。确认冻结SHA6bb7d43997222ffa9c7bb47728c0117d6b8b25712f5df0141f38f06fca78751d，源0102 S-O SHA c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238。旧A—E已核算约17.403进程小时、上限72小时，旧TASK2的59643.306023/60000秒单列，均不构成本轮新研究授权。

工作目录E:/03-nn_Transfer_Func/kiki。GPU工程任务及有限A—E研究已通过PR #1合并到main，合并提交9fe0ff25952cfd88372141ab2d5977eb6a455c3d；不代表科研目标全部成立或预算耗尽。本轮按用户要求交付改进方案后停止，研究训练仍需按新的明确范围与资源登记。本地HEAD a951013和用户index保持，无关改动保留。环境、原始数据和临时模型留本地，关键代码、验证和性能证据上传GitHub。
