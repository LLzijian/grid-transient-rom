# Run3 L2/L3 冻结组件独立实现审查

结论：接受本次冻结分析组件，状态 `analysis_component_accepted_integration_pending`。未发现需要改变方程、科学门或预算的组件阻断；`expected_identity` 的来源引用及调用次数须在runner整体验收落实。此接受不是正式Run3释放，也不代表实际模型获证、谱匹配或Kron已计算。

## 绑定输入与核查范围

| 输入 | SHA256 |
|---|---|
| `experiments/gftd_fit297_analysis_20261107.py` | `2FAA8CDD83A98F71C044E20983B2B08D0F394F76B64C15B24F2C63617DAB929E` |
| `tests/test_gftd_fit297_analysis.py` | `55F1E706C4F25BC04DFBF1526B3A6200B97D5FAE1D4DCA9EB0F6062A3ED0256F` |
| 最终协议 | `CA22743C53FE4FE398960E111838A373F8C4686EAB0974C5DAF6D9C28435062F` |
| `static_identity_audit.json` | `EAC00A21B0AF24B0D7F20A0289D94232AA4F5222568B7162AA93D2BE3DEC2C0C` |
| 本次 `analysis_independent_audit.json` | `0EDA314E2DEBEDB0DB4B3C288FEEC5757433A75AC80ACFA9FD4DA965A50964D3` |

后两文件在 `results/gftd_fit297_20261107/`。同时核对已接受L2舍入/Kron推导、当前核心公式、历史descriptor导出源码和静态证据；未读取仍在实现的引擎/API/runner。父方报告最终12项小型工程资格通过，本次未重跑。独立执行仅15个纯标量Taylor包络与120位独立级数/Decimal比较，全部通过；没有执行实际证书、Kron、ROM/FOM谱、RHS、fit297事件读取、积分或优化。这是解析/源码复核，不是新增独立实验。

## L2：同方程、严格有理证书

- `58–72` 行从各存档因子精确构造K与β，中心按精确有理均值去共同角，保留η舍入偏差。J*K_r、K_r/tau及gain*S_p也保存精确因子积，不用理想κ/λ覆盖已存倒数。与实际Hamiltonian、完整p/r端口相符。
- `75–157` 行核正定储能、严格耗散、活跃图连通及0≤β≤K，固定13个R，使用共同cos正域、exp的有理界、图谱下界、正确的 `2R+1.5δ` 混合界和det/trace完整49维μ。中心梯度门及ρ选择完全model-only；失败为not_certified。裁剪sigmoid的 `1/(1+3^40)` 是合法正下界；本Run实际关闭状态门。
- 点证书核功能体哈希、精确商空间球和同一H的有符号区间能量；sin/cos多项式及38/37阶Lagrange余项、exp的19阶余项合法，负系数和区间乘法处理正确。`V_upper=H_upper+g²/(2μ)` 使用Hc下界0，裕度是能量裕度。ε始终unknown，不把记录σ0或小残差当真实初始化集合证书。
- 整球上界包括δ，二维行和给Hessian谱上界合法。原L2的存在性、局部不变分支及LaSalle前提保留，未把PSD功率审计当吸引证明。`actual_point_count`初始值不是调用计数凭证，runner须单独登记最多594个实际点包络。

## L3：端口网络和局部模态

- `245–387` 行仅解析哈希绑定工作簿，正确按bus1电压/设备Sn换基、bus1侧tap/phase和两端充电叠加；19个PQ使用带符号 `(p0−jq0)/Bus.v0²`，包括保留端负荷。39节点顺序、10机bus30–39、零phase均核验。一次SVD/solve构造Schur，不用逆、伪逆或正则；求解、端口关系与复对称门明确，不要求Hermitian/零行和。没有把PQZ锚点称实际FOM负荷等值。
- `435–535` 行只读既有descriptor/来源/accepted哈希，沿原广义特征值有限判据、0/0、残差与行均衡谱校验；13个参考极点及150有限/549无限/1零/0正计数均核原接受记录。参与度采用left-E-right形式，低overlap不造参与度；完整10ω右向量按ω_s和共同sqrt(M)变换。被拒160维linearization不被调用。
- `538–672` 行L3用已核正交Helmert基，属合法数值商空间基；不用于严格L2，也不改初始化MGS基。保留完整ω/p/a/r，仅去共同θ。Newton最多20更新/每步10线搜索，限定区域、无ridge，RHS平衡残差≤1e-10后才模态分析；两49维Jacobian/谱的差分、候选归属和极点一致性门保留。每模型Newton梯度调用在协议1000内，模态差分196 RHS另计。
- `675` 行以后显式inf全配对，成本与共享helper一致，不按高误差或低MAC过滤，保留所有未匹配。图对应是声明参考状态的耦合/导纳排序与分别归一化权重，不是物理真边FP/FN。runner另附fit的raw/null机制资格，不能把helper的raw_supported或正权直接称恢复合格。

## 静态身份链及整体验收交接

已核静态audit给两shard相同case/signature、GENROU_1…10、ω_s及逐位相同的delta/full_state/algebraic_state摘要；R_N相角仅共同角偏移，投影差为舍入量级。历史导出commit `628c7d917881994d883d68f41f4a21848104603a` 的源码明确比较postfault保存平衡，来源evidence记录全/代数误差0。该保留来源链支持使用同平衡的既有descriptor；descriptor.npz本身未存平衡数组，本次没有重建其平衡或新运行FOM，不能如此声称。

`static_identity_audit.expected_fom_identity`没有 `equilibrium_verification_reference`，而 `fom_modes`要求它非空。root已确认调用方将加入 `static_identity_audit.json#sha256=EAC00…`，且release.asset_sha256强制绑定并核验该文件。**此项是最终集成验收必查，不需改冻结audit或本组件。** 不应只传一组未经绑定的true布尔值。

最终runner还须核：每模型一次13半径证书及最多每合格返回点一次包络；共同Kron一次、共同FOM两QZ一次；每模型有限Newton与两格49维分析一次；未知/失败保留原因，不重算追通过。模型与点哈希、相同质量/单位、raw/null图资格和各调用actual计数也须随结果归档。其他实现来源尚未在本报告接受。

代码决定：保留此分析组件进入整合；模型决定：没有实际证书、匹配或晋升；科学决定：按已审协议继续准备，无新研究资格要求。报告与标量审计至此停写。
