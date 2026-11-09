# GFTD 固定 DP8 候选执行入口独立审查

**接受 44 包完整初始化小试的首次释放。** 全体验证保留独立释放，只有小试新增合格输出且 5 个原成功包均保留资格时才能启动。 [冻结协议](../../campaigns/gftd_initialization_precision_20261109/protocol.md) · [候选采用记录](../../results/gftd_initialization_precision_20261109/candidate_adoption.json)

- **冻结来源与有限工作量。** 入口核对 13 个源码与测试文件、3 个父清单、协议、候选采用、固定初态决策和选择清单的来源链。运行前重新认证父模型与因果包。49 个诊断状态按模型和事件索引去重为 44 包，小试每包调用一次已审查的完整 API，沿用 simple 600 和 smooth 1200 上限。全体验证要求同一来源绑定及运行环境，认证并复用小试模型、诊断与状态原字节，只对其余 550 包初始化。 [释放校验](../../experiments/gftd_initialization_precision_20261109/candidate.py#L63) · [API 审查](api_review.md) · [小试复用](../../experiments/gftd_initialization_precision_20261109/candidate.py#L334)
- **长时积分和参考。** 全体验证从每个合格点持久化的 50 维 float64 状态启动主细积分，复核真实 p₀ 和状态字节。15 s 主细步长为 0.01/0.005 s，记录 1/2/5/8/15 s 的可比前缀与失败时间。新合格点用同一状态做一次 1 s 严格 Radau，与已算出的细步前缀按登记阈值比较。15 s 的后段失败仍保留此前的参考核对。若细步重现竟在 1 s 前中断，当前实现记录参考未决且未尝试。全体释放前应登记该情况为需要对账的技术异常，因为同一固定状态已通过 1 s 细步资格。 [状态校验和长时配对](../../experiments/gftd_initialization_precision_20261109/candidate.py#L236)
- **启动、成本与分母。** 两阶段和逐任务凭据均独占写入。最多 4 个 Windows spawn 工作进程，每进程的 BLAS、OMP、MKL 线程设为 1。逐任务保存实际 RHS、残差、积分尝试与完成数、工作进程 CPU 与耗时，宿主耗时另列。全体阶段新成本与复用小试成本分列。中断先对账任务凭据与已写证据。结果表保留两模型各 297 行，稳定与失稳分别汇总。 [逐任务执行](../../experiments/gftd_initialization_precision_20261109/candidate.py#L289) · [阶段执行和成本](../../experiments/gftd_initialization_precision_20261109/candidate.py#L399)

入口、测试、人工回执的 SHA-256 分别为 `c804a58528fa258d25901fea70fe4fdfae8966e460c6ccc8f64794003fdab92e`、`2ff17d1e3dcf42d4d52a7ba26e3f0a536733a68b31c4dc87d91bcc659cd7b8df` 和 `64189ac6c32f2d17cbd9528c28f6daa02f2838dfde4155ce0943ef2ea06d9b5d`。沿用 5 项人工检查通过回执，其中一项覆盖真实 Windows spawn 工作进程及人工 50 维两阶段流程。本次审查未运行真实数据初始化或积分。 [人工检查回执](../../results/gftd_initialization_precision_20261109/candidate_qualification.json)
