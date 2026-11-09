# M0来源补证：Pm字段与上游tm的含义

2026-10-31，root在真实诊断执行前核查公开上游来源；未运行ANDES，未读取新FOM或val/test。
此补证不改变冻结协议、数据坐标、模型、阈值或部署合同，纳入M0物理来源限制解释。

仓库generate_private_dataset.py第154、301行将Pm映射到GENROU.tm，第386行也记录该映射；
src/kiki/data/private_dataset.py原样保留raw[Pm]为Pm_raw；现有train再减冻结base.pm_star。
因此现有p是这个既定数值字段的偏差，不能仅凭Pm命名独立确定其物理功率含义。

ANDES v2.0.0的[GENBase源码](https://raw.githubusercontent.com/CURENT/andes/v2.0.0/andes/models/synchronous/genbase.py)
把tm标为“mechanical torque”；其转速方程由tm、te和阻尼组成。
[GENROU源码](https://raw.githubusercontent.com/CURENT/andes/v2.0.0/andes/models/synchronous/genrou.py)
使用GENBase与Flux0；[同版本官方模型说明](https://docs.andes.app/en/v2.0.0/reference/models/SynGen.html)
也将tm与电网有功注入Pe分列。上述页面均于2026-10-31实际打开核查。

**可支持的结论：** 上游字段有明确的转矩标注，仓库保留了该字段的数值。
若将其称为真实机械功率或作SI功率账，须另给该模型近似下的物理解释、标幺基准及变换依据。
**尚不支持的结论：** 不能仅凭标注认定当前全部数值预测无效，也不能据此自动乘转速或容量因子修正数据。
GENROU的模型近似及本机实际版本/生成环境还应与已冻结来源绑定。

当前Run继续按原p坐标诊断；公开数值的pu仅沿冻结数据的标幺坐标约定，不独立认证物理功率单位。
代码中的physical_unit_provenance仍保留来源限制。任何未来目标或坐标修订须先完成单独设计与原部署合同核对。
