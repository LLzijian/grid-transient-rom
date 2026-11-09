# Run3 L3：bus-terminal Kron 组装约定核查

结论：已有静态资产足以定义一个可复现的 **节点端电压参考**。建议 root 在 Run3 协议中采用下述唯一主参考：`bus_terminal_line_shunt_pqz_casev0_s100_v1`，保留 bus 30–39、消去 bus 1–29，以 Line+Shunt 加全部 PQ 的固定阻抗构造复数 Y，再作 Schur 消元。本页是待采纳的计算定义，尚未组装矩阵、求解潮流或计算 Kron；不将其称为原 FOM 的精确动态等值或内部电势网络。

输入工作簿为 `.tmp/gftd_ieee39_reference_20261107/ieee39_full.xlsx`，SHA256 `9C2048DC94201EE48FFE816DE65D53831FA1F4B0E7FDF32F24C50F1017EDCFE5`，本次重新校验匹配。来源和机组顺序已在 `gftd_fit297_20261107_physical_reference.md`（SHA256 `6E332F85D2906A45CB7175CC8FBAD1F12CFB9B38CC50AE3525894A8B7DA9D9F3`）记录，本页仅补组装合同。没有读取 fit297 事件、训练、仿真或改变 Run2 的冻结源。

## 1. 官方实现、版本和换基

核对 CURENT/andes `v2.0.0`，tag 对应 commit [`eda5163c9ee8d19945a1dd5d1771fec5da608c27`](https://github.com/CURENT/andes/commit/eda5163c9ee8d19945a1dd5d1771fec5da608c27)。以下 SHA 为 Git blob SHA-1，区分于工作簿 SHA256：

| 官方源码 | Git blob SHA-1 | 核对对象 |
|---|---|---|
| [Line](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/line/line.py#L251) | `df0afa0d38ce26978caee043ddc6ffca42d386ee` | `build_ybus`，tap/phase 方向 |
| [Shunt](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/shunt/shunt.py) | `4b46e69a35c01f9cd20a62ef96dcd0296bab3bf7` | 导纳正负号 |
| [System facade](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/system/facade.py#L771) | `e9e5a2fa713cb6dc22f1df87eaf099c5fc050274` | `calc_pu_coeff`、`build_ybus` |
| [系统默认配置](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/system/config_runtime.py#L239) | `3ac6da80610a82b400d2814db257e40a970974bf` | mva=100、freq=60 |
| [参数换基](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/core/param.py#L552) | `94a1cb95bef9080793ede6834183b41f105ddbe3` | `v=vin*pu_coeff` |
| [PQ](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/static/pq.py) | `d6b873f168651e4dd7c467494bf2776b0105081e` | p0/q0 已在系统功率基准；TDS 的 ZIP 转换 |
| [Bus](https://github.com/CURENT/andes/blob/eda5163c9ee8d19945a1dd5d1771fec5da608c27/andes/models/bus.py) | `dd04446de91d02111955a8e5ddb92ff7abdb50fa` | Vn、v0/a0 为输入初值 |

主参考显式固定 `Sbase=100 MVA, fbase=60 Hz`、各 bus 的电压基值为工作簿 `Bus.Vn`（kV）。这是本参考的定义；本次没有恢复并验证历史 FOM 的 runtime 配置。ANDES 对接 bus 的设备取 `Vb=Bus.Vn(bus)`；Line 取 **bus1** 的 Vb 和设备 Vn1。令 `Zn=Vn²/Sn, Zb=Vb²/Sbase`，则阻抗字段乘 `Zn/Zb`，导纳字段乘 `Zb/Zn`；标有 power 的参数才乘 `Sn/Sbase`。本 case 的全部 Line/Shunt Sn=100，额定电压与所接母线基值相符，故这些 z/y 换基系数均为 1。不能拿 GENROU 的不同 Sn 再换一次 Line/PQ；PQ.p0/q0 已是系统基准标幺值。

Vn2 与 bus2 的额定电压也逐行匹配；ANDES 的该 Line 换基实现并不额外把 `Vn1/Vn2` 乘进 tap。本参考按此实现，不能在已经按母线电压基值表达的标幺电压间重复加入额定变比。b/g 是给定频率下的标幺导纳，不再乘 2πf。

## 2. 唯一组装定义

按 bus idx 1,…,39 建显式索引，不使用 sheet 行号作节点身份；取工作簿 u，全部 39 bus、46 Line、2 Shunt 和 19 PQ 均在线。所有支路（包括 `trans=1` 的 12 条）按同一公式叠加，不以 trans 决定是否采用 tap。

对一条 `f=bus1,t=bus2`，以下字段均已换系统基准：

`ys=u/(r+jx)`，`yh=u*(g+jb)/2`，`y1=u*(g1+jb1)`，`y2=u*(g2+jb2)`，`m=tap*exp(j*phi)`。

`Yff += (ys+y1+yh)/tap²`，`Yft += -ys/conj(m)`，`Ytf += -ys/m`，`Ytt += ys+y2+yh`。

tap 位于 **bus1 侧**，phi 单位 rad。本 case phi 全零；Line_36 是 `31→6,tap=.9`，不能统一把 tap 搬到高压侧，也不能任意交换端点而不变换模型。Shunt 直接给对应对角元加 `u*(g+jb)`，正 b 表示容性。公式对应官方 `Line.build_ybus`，不向 r/x 添加 epsilon；同文件 DAE 服务 `yhk` 带 `1e-8` 正则，因此不能把此矩阵进一步称为逐位等同实际 DAE 方程。

**固定 PQ 假设：** 对每条负荷取锚点 `Vref=Bus.v0(bus)>0`，加入 `Yload=u*(p0-j*q0)/Vref²`，包括保留端的 bus31/39 负荷；同 bus 项求和。p/q 为消耗为正，bus24 的负 q0 必须保留，不能取绝对值。这保证该参考阻抗在声明锚点电压幅值处消耗输入 p0/q0；Bus.v0 在这里只是静态输入锚点，没有被当作已核实潮流或动态平衡。

官方 `System.build_ybus` 只汇总 `flags.ybus` 模型；PQ 本身不自动加入。官方 TDS 默认 p2z=q2z=1，但实际导纳由求解后的 `Ppf,Qpf,v0` 得到，且配置可改成 ZIP。因此本页的固定 PQ 添加是**明确的参考假设**，不是声称复现历史 FOM 的负荷转换。若未来要改用经核验的稳态电压/功率，必须前瞻另记参考版本，不能在看图比较后切换。本主参考不把恒功率负荷直接塞入线性 Y，不添加 PV/Slack 或 GENROU 的源内阻。

## 3. 消元、实现核验与比较范围

冻结 `R=[30,…,39], E=[1,…,29]`，与 GENROU_1,…,10 一一对应。使用复数线性求解 `T=solve(Y_EE,Y_ER)`，定义 `Yred=Y_RR-Y_RE*T`，不显式求逆、不加任意对角正则或伪逆替代。这是固定阻抗负荷已吸收入 Y、消去节点外部电流为零的端口关系 `I_R=Yred V_R`。若仍保留非零电流源，则还存在消元后的仿射项，不能只给同一个 Yred 充当完整关系。

计算前/后应核对状态/映射、换基系数、tap 方向和所有 p/q 符号；保存 Y_EE 的奇异谱/条件数、线性求解相对残差与 Schur 关系残差。本 case phi=0、实 tap 的 Y 应为**复对称**（转置），不要求 Hermitian；有对地并联项，不强制行和为零。若块不可逆或残差不合格，返回 unknown，不静默删 shunt、负荷或 epsilon“修好”。本页未计算实际条件数、秩、稠密程度或任何这些残差。

Kron 消元通常产生稠密端口耦合；发现的稀疏有效图描述有限观测下的自治 ROM，二者不具有相同的真边含义。L3 可在同机组顺序下，对 45 条无向边的 `|Yred_ij|` 与 ROM 的声明参考状态有效耦合报告排序关联、归一化权重及边作用分布；须注明一个是端电压导纳、一个是转子相角有效作用。损耗 G、端电压幅值和相角工作点未映射时，不将原始复导纳或其虚部直接称为同单位 k。不得报告“物理真边 FP/FN”或据不一致否定有效图。真实 rotor-angle 模态比较还需统一工作点、角度/速度与电压的映射，不能由此 Yred 自动得到。

内部电势 Kron 暂不做：需要另冻选用暂态/次暂态模型、ra 与 x'd/x'q 的基准及取法、内部 E 的定义/稳态初始化、负荷处理和控制器闭合。现有 GENROU 列本身不等于恒内部电势模型，bus-terminal 相角也不等于 rotor δ。此缺口不阻止上述 report-only 端口参考，但不支持内部 EMF 或完整 FOM 结构恢复主张。报告至此停写，待 root 采纳具体 Run3 参考合同。
