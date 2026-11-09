# 记忆转运模型：独立静态审查

2026-10-29。实际模型 `gpt-6.1-sol`，独立审查者，未实现本模块。只读新model/test及必要父field、线性化、缓存和初始化定义；未编译、运行测试、读取真实源参数或事件，未计算源平面或数值。root保留实际执行与采用权。

**未发现阻断有限CPU数学资格的决定性方程/坐标错误。** root可在既定归档与释放条件满足后，只执行明确选择的CPU资格；此结论不是数学检查已经通过，更不放行真实源谱或CUDA/训练。正式源/检查点接线尚需下列两个小项闭合。

## 静态已核

- model:141–149构造对称Q、P2及skew=sK，完整generator=Q+dP2+sK；:162–176未来场使用r@Aᵀ，:219–229历史仿射列使用−A@affine，矩阵次序对非对称S正确。:192–196在线性化内部π坐标r块覆盖为同一−generator，没有残留对称历史近似。p准确块、F/q/QR和原末点信息权限保持。
- :178–190的J/R明示内部π=Mω，`J_rr=−κsK`，R_rr=κ(Q+dP2)；乘原∇H得到所需−sKr和耗散项。继承field/initialize仍对外输出ω，不把J直接作用于外ω。完整十维ω保留，无额外p-r交换或端口。
- :93–94的s/d是直接参数；:247–256在真实step之后投影d与原边，d=0前向和可行单侧导数合法，没有softplus/ReLU死边。s可有符号。A/S/P的零s/d构成同一源的嵌入；测试:197–224按1e−12初始化/轨迹门检查，没有假称逐位复现旧初始化器。
- :138–149的缓存继承父版本、grad模式、设备、buffer/plane与requires_grad等stamp检查，projection/load/train/to清缓存；完整generator保留参数图，不detach s/d/原参数。history(reference=True)与field_reference用于同公式未缓存支路。局部/完整J allgrad检查定义分别沿已登记门；实际是否数值通过须执行后记录。
- :266–314使用独立format保存variant、s/d、plane及来源、history_dt、state/gradient布局，加载核固定buffer/config/plane一致，拒旧格式吞掉S，from_a_export只按冻结源SHA拷原A参数。variant/mode/gate_mode不能任意切换。不加载旧Adam；实际Adam/RNG恢复仍属于另行真实QA。
- :30–80的公开prepare_plane先独占目录及attempt，再读取冻结源及做唯一9D Helmert eigh；hash失败/异常保留结果，cutoff未过为UNRESOLVED，不重挑平面。argmax符号规则与方案一致，没有第二次谱分解。导入仅定义，不自动选面或发起计算；必须由root/controller绑定唯一事务目录后调用，换目录再调用不是额外授权。测试的选面检查仅mock人工export上的eigh，不会计算真实A96平面。

## 正式源/检查点前需闭合

1. **补齐已登记的选面残差记录。** model:70–80结果已有9特征值、gap/cutoff、u/v和source身份，但没有协议要求的向量/正交残差。应在同一次既有分解中记录这些残差及实际来源，不加第二次谱、择面或新门。此记录缺口不影响人工CPU套件，真实选面事务前须落实。
2. **细历史模型不可误作主部署检查点。** 新:266–277的export覆盖了父类对参考历史的导出限制，目前构造history_dt=.00125后也可导出/载入。注册的主历史为.0025、细.00125用于资格。最小处理可在model export及对应主候选load拒绝参考历史，或在后续正式checkpoint/部署入口明确且独立核强制主.0025；不要只依赖调用习惯。可继续通过history_dt关键字执行细路径而不改变主模型配置，不需改数学方法。当前CPU数学不导出细历史模型，故不必为这项等待额外真实计算。

## 执行必须分两次显式选择

本文件测试:360–381的actual CUDA合成配对在没有设备时会skip，而有设备时默认全文件运行会立即包含它。root应先明确选择CPU数学套件、排除该CUDA函数；只有CPU结果合格且对应下一阶段明确释放后，再单独选择实际CUDA合成配对。skip记未执行/未决，不能把全文件CPU PASS含skip写成CPU/CUDA全通过，也不能因设备可用在CPU资格未完成前自动运行CUDA。

CPU选择包含已登记固定人工M3、Helmert前两列、A/S/P非零s/d、独立负时间块参考、DOP853参考与实际真RK4因果重建、缓存/全梯度、零嵌入、导出、投影及mock选面定义；没有真实源参数或事件。不得把这份静态报告当CPU suite输出。GPU合成检查也不替代真实源fit297、六次12×8性能、原物理数值门及恢复QA108。

## 本次源身份

| 文件 | 实际SHA256 |
|---|---|
| experiments/memory_transport_model_20261029.py | `188f5d6bc33b0c1b26c6ed9ebae95e14784fddc355b362ab90de25e87b6754ce` |
| tests/test_memory_transport_model_20261029.py | `fdd48f5d2eadd22bdd0fd7642b2b4d012f2a40291138ca807c0f0e1020be20d1` |

只写本报告，原科学门、协议、代码及结果未改。后续真实源/正式检查点小项由root选择最小实现并归档，不扩成通用认证框架。

## 两项小修的局部复核

2026-10-29，实际 `gpt-6.1-sol`，仅只读原两处及新增测试定义。**局部PASS，选面记录与主历史导出限制均已闭合，无此次范围内残余项。** 原发现、初版SHA及“尚未执行数学资格”的事实保留。

- 新model:80–89由同一个实际Q、所选plane及原特征值计算正交Frobenius、列范数、零和与各特征向量残差，另按原固定cutoff尺度记录归一化残差及单位。未增加eigh或新选择门；cutoff未过时:75输出plane_checks=None，没有伪造合格残差。新增mock定义用不吻合实际Q的特征对核残差确实测量而非硬编码0，仅为定义，未运行。
- 新model:275–277在写文件前硬拒history_dt非.0025；:290–294加载入口也硬拒参考history。细.00125仍可通过关键字或参考对象计算，不能导出/恢复为候选。新增测试 `test_reference_history_can_compute_but_cannot_export_or_restore_checkpoint` 对A/S/P分别定义细路径可计算、细export无文件、主checkpoint可恢复、篡改为细history拒载的检查，没有改变原数学门或主/细精度。

实际新SHA256：model `c9040bd65026edd0939b6c1046e21205388ffe1d2d5d69752d67144416101494`；test `0e043c1200aed9de6d11e21dc57900c23ec6922dd75d71ea649e9dc39ba9989a`。

root归档核验后可按已定授权仅选择CPU数学套件，显式排除 `test_actual_CUDA_field_initialize_short_complete_J_all_gradients`。本次没有执行测试/编译/源谱/数据；CPU合格后CUDA仍须分别显式授权，skip不作PASS，真实资格与训练均未由本报告释放。
