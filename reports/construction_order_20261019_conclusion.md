# 构造顺序工程：检查中断，吞吐未决

唯一CUDA等价检查在新增检查代码的有限值校验处中断；benchmark未启动，构造函数不采用、不合并main。原默认入口与科学模型保持，源及负结果保留在codex/construction-order-20261019。

错误发生于tests/test_closure_construction_order_20261019.py:351：将CUDA参数与含CPU Adam step的状态一起交给原all_finite，原工具对torch.isfinite返回值直接stack，因设备不同抛RuntimeError。该新增调用方式由本轮检查引入。独立静态审查未发现它；root执行后保持代码冻结，遵守无修正重跑约定。

执行已到full8初值、短/5秒目标及梯度、长更新状态比较之后的检查位置；没有完整通过记录。不能把此前未失败的代码路径视为全部验收，也不能据此判断七项结构、恢复或吞吐。没有证明构造顺序无效；工程采用条件尚未成立。旧gate及18.4908032秒最大值不变，P2仍无准入。

源冻结归档901dcb60ed9270c06fe2138f5ab55cc04c99f878。构造函数SHA256 8727f43f0967cbb2e5b36092f553b8ac75923902c6f65c861a051ac5bba430d5；检查/测时SHA256 0812c459609dd1f89a0765aec19aee4c5863f18ffef628d94db9ca730cac0e51。失败后20份绑定源哈希全部未变。无真实train、val或test访问，TASK4继续暂停。

唯一命令：`.venv-gpu/Scripts/python.exe tests/test_closure_construction_order_20261019.py --mode check --output private_data/closure_mechanism_20261019/P0/construction_order01/check.json`。该试验关闭，命令供复查已发生执行，不授予重跑。复现需要登记私有S-O权重；原始产物在上述本地目录，紧凑证据与各文件哈希见[summary.json](../results/construction_order_20261019/summary.json)。实际worker72860、launcher86104、监督82476，2026-10-19 17:46:25至17:47:42，退出码1、未超时，已核对进程退出。

完整启动及子进程一次计76.9246693进程秒；Python内部50.7492878不另加。1800分配按实际76.9246693+支持保守300关闭，未用执行1423.0753307明确撤销，benchmark费用0。P0剩至少11376.4913043，父259200上限下剩至少231023.9964663，修复预留14065.4147849及P1剩3982.0903771不变。旧900支持仍扣留，不移用旧账本或阶段额度。

下一决定由实际Astra6有限分析剩余合法完整路径，不以本次检查失败自动宣布任务完成或全局阻塞。不得重开本次构造顺序检查、借预留追加其修复/测速，亦不得把身份接口PR #8的工程接受解释为科学采用。
