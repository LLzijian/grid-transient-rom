# fit297物理参考：找回case的静态核验

已找回与原数据manifest完全一致的case，可为Run3冻结静态参考提供输入；尚未构造Ybus、节点电压Kron或内部电势Kron。来源为[CURENT/andes官方工作簿](https://github.com/CURENT/andes/blob/master/andes/cases/ieee39/ieee39_full.xlsx)，[固定Git blob](https://api.github.com/repos/CURENT/andes/git/blobs/f49523f01b15d833c4376f020f4ba4846478e8e3)。本地缓存`.tmp/gftd_ieee39_reference_20261107/ieee39_full.xlsx`，38,449 bytes；SHA256 `9C2048DC94201EE48FFE816DE65D53831FA1F4B0E7FDF32F24C50F1017EDCFE5`，与两shard原case_sha256一致；来源记录为同目录retrieval.json（2026-11-07）。本次用标准库ZIP/XML只读解析，无新增依赖。

GENROU表头包含idx/u/bus/gen/Sn/Vn/fn/D/M/ra/xl/xd1/xq1/xd2/xq2/xd/xq/Td10/Td20/Tq10/Tq20等。以下均为原表值，未换算电抗或功率基值；10台均u=1、fn=60，gen关联ID依次1–10。与旧项目`data/kiki_ieee39_v1/manifest.json`中postfault/expansion的delta state_mapping顺序逐项一致：

|机组|bus|额定Sn|额定Vn|xd1=xq1|xd2=xq2|
|---|---:|---:|---:|---:|---:|
|GENROU_1|30|1040|34.5|.31|.00792|
|GENROU_2|31|836|34.5|.697|.29779|
|GENROU_3|32|843.7|21|.531|.02566|
|GENROU_4|33|1174.8|21|.436|.00803|
|GENROU_5|34|1080.2|15.5|1.32|.00571|
|GENROU_6|35|1085.7|15.5|.5|.46538|
|GENROU_7|36|1025.2|12.5|.49|.02141|
|GENROU_8|37|970.2|12.5|.57|.06076|
|GENROU_9|38|1684.1|34.5|.57|.01174|
|GENROU_10|39|1199|345|.06|.0026|

工作簿共14表：Bus、PQ、PV、Slack、Shunt、Line、Area、GENROU、TGOV1N、IEEEX1、IEEEST、BusFreq、ACEc、Toggler。以下行数按非空idx计算，排除PV格式空行：

|静态对象|有效记录/状态|建网或参考所需字段|
|---|---|---|
|Bus|39，全部u=1|idx、Vn、v0、a0、area/zone；idx为1–39|
|Line|46，全部u=1；34普通支路+12 trans=1|bus1/bus2、Sn/fn/Vn1/Vn2、r/x、b/g、b1/g1/b2/g2、trans/tap/phi；tap范围.9–1.07、phi全0|
|Transformer|无单独sheet|12台变压器编码在Line，不能遗漏其tap和两侧电压基值|
|Shunt|2，全部u=1；bus4/5|Sn/Vn、g/b、fn；g均0、b分别1/2|
|PQ|19，全部u=1|bus/Vn、p0/q0、vmax/vmin；含负q0，不能擅自取绝对值|
|PV/Slack|9/1，全部u=1；Slack在bus39|idx、bus、Sn/Vn、p0/q0、v0及限值、ra/xs；并非已导出的内部电势|

其余控制器表均有有效记录；唯一Toggler为u=0、GENROU_1、t=12。静态u状态与case字节被核对，不代表执行了事件或潮流。

Run3还须前瞻冻结三类假设：①节点电压参考的保留/消去节点、系统基值与变压器方向/分接约定，以及PQ按恒功率还是在选定稳态转恒阻抗；原case不能自动给出某个负荷等效的Kron。②内部电势参考需另选暂态/次暂态表示、ra及xd1/xq1或xd2/xq2的取法与换基、E的定义及稳态初始化；本表虽两轴对应值相等，不能据此把完整GENROU动态无条件替成恒内电势模型。③Bus.v0/a0及PV/Slack输入不是本次已验证的潮流/动态平衡，不能代替原全局equilibrium元数据；L3参考与拟合模型须统一机组顺序、坐标、频率/阻尼和观测映射，仍为report-only。

本次不读取事件轨迹或val/test，不运行ANDES、潮流、FOM、Kron消元、训练或模态计算。先前“未找到case”的历史定位记录不改写；本页记录新增的哈希匹配静态资产事实，至此停写。
