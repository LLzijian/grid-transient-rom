# GFTD 固定 DP8 候选 API 独立实现审查

**接受 API 实现。** 候选通过显式配置选择固定 DP8，同一八阶公式贯穿历史拟合、数值资格与公开预测。旧配置保持原 RK4 路径。真实候选小试另按执行入口审查并释放。

- **配置与存取。** 包装配置绑定协议哈希、SciPy 八阶主公式系数哈希及原 fit297 配置。加载、保存和初始化校验嵌套配置与模型指纹，拒绝 DP8 配置和模型模式、初始化方式或父初始化器不匹配。公开入口只收 11 点因果历史与末点 p。 [配置校验](../../src/kiki/models/gftd_fitting.py#L77) · [模型入口与存取](../../src/kiki/models/gftd_kuramoto.py#L450)
- **两阶段和原六门。** simple 从零起点开始，上限 600 次残差。通过原门后，smooth 从原可行点开始，上限 1200 次残差。嵌套 fit297 配置继续提供原历史尺度与记忆状态尺度，末点 p 逐值硬锚。历史、往返和未来资格沿用原步长、原投影和六道门槛。DP8 失败保留原阶段失败语义，累计已实际调用的 RHS。每个完成步用 12 次，失败步按实际进入的次数计。 [积分计数](../../src/kiki/models/gftd_fitting.py#L137) · [资格和阶段选择](../../src/kiki/models/gftd_fitting.py#L288)
- **公开预测及旧配置。** 公开预测校验从零开始的均匀 0.01 或 0.005 s 网格，并以实际固定步长推进已初始化状态。数值失败直接传出。旧 RK4 的积分算式、选点、门槛与计数字段沿用原路径。候选主细配对执行时应复用同一个合格状态，预测成本单独记录。 [公开预测](../../src/kiki/models/gftd_kuramoto.py#L511) · [候选 API 测试](../../tests/test_gftd_precision_api.py)

| 核对文件 | SHA-256 |
|---|---|
| `gftd_fitting.py` | `f45656e1d7224a1a745ada6c9d2191f9f9af01153d891f17c35b629a9eb5c973` |
| `gftd_kuramoto.py` | `124c9df04dc7eaba7e3ddc435e7646a818c1a01edd5284cd8743c63d124220d1` |
| `test_gftd_precision_api.py` | `49dafbc9703fb77489271d2eff37582fbb87efd8fcc5614c8b00473efc71d744` |

联合人工回归 111 项和 9 个子测试通过，新 API 单独测试 10 项通过。独立审查核对源码差异和人工测试回执，未运行真实模型。 [API 工程检查回执](../../results/gftd_initialization_precision_20261109/api_qualification.json)
