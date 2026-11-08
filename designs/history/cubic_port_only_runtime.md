# 三次动量端口运行模块

`experiments/cubic_port_only_runtime_20261021.py`提供可选P8表示：原七项储能及端口保持，只增加共享有符号系数b的三次动量端口。它不改变默认模型，也没有已训练P8端点。

方程为H_P=H_N、u_P=u_N+b*q^3，新增dot(pi)=b*q^3、dot(omega)=b*q^3/M。储能率贡献sum(omega*b*q^3)使用模型H-unit/s，无预定符号。b直接从0开始，归closure参数组，不作非负投影；无四次储能项或a。外部u替换完整自治端口时，也去掉b项。q保持原潜坐标含义。

使用build_P(N7_source)构造，或PRuntime.from_export(path)恢复。保留encode/field/rollout/export接口，按term_key定位b；通用表达式计算与独立NumPy解释器可读取相同导出。每次优化图内合并固定矩阵乘积并保留完整梯度，结束、异常、load/device转换清空缓存；活动图内拒绝导出。导出包含标准表达式和系数双副本，不增加模式。

2026-10-21的唯一研究QA共42项，**41通过、1失败**。其中本模块的21个CPU/CUDA检查全部通过：非单位惯性/外部输入、通用/合并/NumPy场、完整5s原目标和raw梯度/clip/Adam、连续更新、非空Adam与CPU/CUDA/NumPy RNG文件恢复、零b嵌入和可学习性、完整机器重编号、共同角、一次初始化/真实n=1、缓存/非有限与导出检查。它们是同一次QA的通过子集，没有重跑为另一次“全通过”。field容差rtol2e-11/atol1e-11，loss2e-10、grad/Adam2e-9。

失败位于研究runner的合成身份测试：测试用变量名'0'等构造七项AST，违反合法语法，被term_key正确拒绝。按事前无QA02规则，本轮成本测量、三个1024训练与正式评价均未启动；尚未验证真实三O512和六N/Q的完整同源合同，没有预测收益或模型采用结论。相关runner、失败测试及原native证据保存在归档分支，不作为本模块通过检查的一部分集成。

复现资料在`results/cubic_port_only_20261021/`及本轮归档分支`codex/cubic-port-only-20261021`，科学源5c909195fb7a5957bc0a8ea91fcd45340063a1aa。原生费用40.8240817秒，pytest33.70秒是其子集。使用Torch2.13.0+cu130/RTX4060Ti、float64；这些结果不构成真实训练速度声明。

runtime测试依赖已经冻结的合成资产：本地`private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01`，通过FUSED_CLOSURE_FIXTURE显式指定；manifest SHA0a1f2463693b0a5227c8f39e725a4e8891af0300dfccdfde705d66e346ee9816、合成data SHA876f05203ba4ebbf0b503e0fa8cac077e22d1f37a779d95f1feca2047f5bc04f。源/命令/绑定及摘要公开，NPZ/RNG本地保留并记录哈希，main不承诺自足CI。本次失败研究的唯一QA不授权再次运行或训练；独立复现/后续研究须另有范围和预算。
