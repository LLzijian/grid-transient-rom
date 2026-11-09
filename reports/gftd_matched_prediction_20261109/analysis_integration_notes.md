# 最终模型的 L2/L3 接入说明

新分析模块沿用原静态证书、有限平衡求解和模态匹配接口。有界模型用独立的导数界和真实核有理包络。两个分支都对最终模型及其正式返回点出具结果。当前交付范围为隔离源码、固定人工检查用例和静态语法检查。

## L2 证书与计数

- `l2_certificate(functional, delta_star)` 每模型检查一次固定 13 半径。静态 `exp_v1` 要求 `beta=0`，保留原 `h=R*G_K` 及点能量算法。有界核按精确的正耦合边与正调制边分别取界，在原半径库中过滤 `b>1/4`，幅值 Hessian 上界含 `1+2b`。模型证书使用精确有理计算，Taylor 调用为零。
- `l2_point(certificate, functional, state, taylor_budget=budget)` 每个正式合格返回点调用一次。先核验核身份、模型指纹和严格球门，再用 `H_upper+g²/(2mu)` 计入舍入中心与真实极小点的差。有界核先包络 `exp(2h)`，经有理单调映射得到 `tanh(h)` 区间，再包络外层指数。向外量化固定为 `2^-128`，空调制边集只执行三角包络。所有成功与正常失败结果保留 `taylor_calls`，误差球半径 `epsilon` 保持 null。
- 调用方为两模型科学点分析共用 `TaylorBudget(160380)`，并按阶段回执限制两次模型证书、594 点及每正式点一次调用。模块对单点另限 270 次。尝试次数在实际 Taylor 调用前计入，达到额度后返回 `point_envelope_failed` 及明细，已用次数不归还。人工检查使用独立额度，可记录 `TaylorBudget.interval` 的计数差。
- 协议第 5 节的固定人工半径示例写作 `cap=1/2`。按第 7 节冻结公式，`A0=0、G_K=G_beta=1` 得 `min(1,7/15,1/2)=7/15`。用例断言 `7/15`，第 0 个半径仍被过滤，第 1 个仍通过。公式、半径库和次数上限均按第 7 节执行。

## L3 平衡、模态与缓存

- `rom_modes(functional, delta_star, certificate=None)` 沿用最多 20 步 Newton、每步最多 10 次回溯及两档差分。返回实际最终平衡及 `equilibrium_a`。每模型至多 980 次独立梯度、217 次 RHS、2 次 eig。每次尝试在调用前计数，失败结果保留 `counters`。
- `graph_correspondence(kron, functional, delta_star)` 与 `match_modes(fom, rom)` 接收由运行方按释放清单核验来源与 SHA 的缓存对象，保持原参考相角刚度和一对一匹配口径。原分析逻辑来自 SHA `2faa8cdd83a98f71c044e20983b2b08d0f394f76b64c15b24f2c63617dab929e`，新模块从公共源码导入共享模态匹配函数。

固定用例见 [tests/test_gftd_matched_analysis.py](../../tests/test_gftd_matched_analysis.py)。模型和测试仍待释放后的独立实现审查与人工执行资格。
