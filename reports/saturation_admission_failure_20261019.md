# 饱和入口首次启动：零更新的身份检查失败

首次启动在任何优化器更新前被严格身份检查拒绝，科学比较尚未开始。三份共享fixture在22.4308041秒内生成并核对原哈希、尺度与数值资格，六项均无检查点、无512端点、无模型val评分。此结果不评价初始化方案或饱和方程的能力。

失败来自结构描述中ParameterList.extra_repr包含设备位置：实际模型在cuda:0，参考在CPU，设备文字差异被当作结构差异。一次CPU只读诊断确认原toy与缓存source重建结构完全相等；没有改文件、训练或绕过检查。新入口原SHA256 fbb3e446a748cfb1cace69e342aef0614f99e5a7411fcf9f9a22c93f7b3e5916保留。原六项QA和独立审查没有覆盖此跨设备描述差异。

实际运行PID79820已退出。收据较大值22.79197399999248秒记一次；原4800分配（4000执行、800支持）仍用于当前实施阶段，执行剩3977.2080260000075秒，不另加预算。QA7.2187849秒、诊断2.4700098秒计原800支持；该支持仍保守保留，不能转换训练。未启动重试，未延长已关闭P0修复次数。

已实际调用新gpt-6-astra只读制定准入恢复方案，需前瞻记录、必要检查和源码归档后才可再次启动。不得将此次CLOSED作业改标运行或中断，不覆盖run01。原历史源码、默认模型、原test关闭及TASK4暂停保持。通用运行模块不集成，本轮实现暂不合main。

复现来源与全部失败栈、费用、缓存manifest哈希见results/closure_saturation_pair_20261019/admission_failure_summary.json。本地private_data/closure_mechanism_20261019/P1/history_residual/saturation_run01保留原source/observations/metadata、日志和原始state/receipt；缓存观测不上传。启动命令为`.venv-gpu/Scripts/python.exe experiments/closure_saturation_pair_20261019.py --output private_data/closure_mechanism_20261019/P1/history_residual/saturation_run01 --device cuda`，既有输出禁止覆盖重跑。
