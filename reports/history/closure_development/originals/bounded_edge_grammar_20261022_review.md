# 增长受控边语法：独立增长与设计挑战

本报告由未参与本轮方案/实现的代理撰写。只读取实际Astra6方案、root协议与实际数值/compiler源码；不运行测试、数据生成、训练、评价、torch.load、Git或联网，不递归委派，仅写本文件。科学采用由root决定。

## 真实编译器的增长结论

**本轮实际筛选的至多线性增长论证成立，未发现中心化引入q×角差的反例。** 结论限定为本协议的固定toy40骨架、正质量/尺度、有限固定参数、edge候选且表达式为qi/qj总次数不超过1的有限多项式，候选集合再要求次数恰1、含s/c和非零mechanism_signature。下述实现静态核对确认筛选保持这些前提；实际冻结池仍由唯一syntax-freeze产生。

关键区别是实际feature与能量中心化。`symbolic_closure_rom.py:_torch_features`中的

`s = sin(phi+eta) − sin(phi)`，`c = cos(phi+eta) − cos(phi)`，`eta = D delta`。

它们本身有界，角度导数也有界；这里没有把s/c定义成扣除了角度线性切线的函数。边对称化只交换端点、令s变号、保留c，不提高q次数。

将对称后的合法多项式写成

`F(qi,qj,s,c) = b(s,c) + qi*a(s,c) + qj*d(s,c)`。

由于s/c取值位于固定紧集，a、b、d及其s/c一阶导数有有限上界。于是物理角度梯度至多为常数加常数乘||q||，q梯度有界；表达式不含pi/pj，所以p梯度为0。

`_torch_feature_eval`对**整个对称标量表达式**扣除原点值和原点feature梯度乘features。扣除q方向项只增加常数q梯度；扣除s/c方向项的梯度仍有界。`_torch_term_value_gradient`另加的物理cos切线修正为

`a0 * (c + sin(phi)*eta)`，`a0 = partial_c F(0,0,0,0)`。

此a0是在q=0处得到的固定常数，来自q零次部分，不能变成qi或qj的函数。因此它给能量增加的是有界项加角差线性项，给角度梯度增加的是`a0*(-sin(phi+eta)+sin(phi))*D`，仍有界；不会产生q乘eta。不能把这个中心化能量中的线性角差误认为q相关三角feature自身无界。

`energy_and_gradient`的seeded edge_sine/edge_cosine快速分支直接实现同类q×有界三角项的梯度，符合上述界；非seeded表达式走刚核对的通用编译分支。`TrainingClosureROM.field`在通用分支调用该实际梯度，在快速分支使用相同s/c及导数，不存在另一个未审查的超线性计算定义。

## 骨架、矩阵、端口及初始化

`toy_source`为10节点环图、40状态、质量1，基础StructuredROM没有额外storage/exchange/dissipation/ports specs。其基础能量的角度项确含线性eta补偿，但角度梯度是`(sin(phi+eta)−sin(phi))*coupling*D`，有界；动量、p和保留q二次储能的梯度为线性项。

LocalMemoryROM只启用template_mask的3、5：3改变pi-q常数交换矩阵，5增加q方向常数耗散矩阵，不把状态乘进J/R。Symbolic转换将旧H前三项及local_port组7禁用；保留q二次项。其余允许的J/R归一化方向也是有限固定参数的常数矩阵，本轮未新增这些组。质量与接口变换是固定正有限线性变换。

本轮terms全为edge，Symbolic端口输出为0，外部u不参与自主预测。即使列举既有六个seeded端口原子，w、p、q、nq为线性项，es、ec为有界三角汇总；有限常数线性组合也至多线性。此论证**不覆盖任意generic port多项式**，例如额外启用q的高次端口会破坏前提；实现应保持edge-only池及原端口屏蔽，不能据此声称生产语法的所有合法port都受同一增长界保护。

CausalHistoryResidual的GRU和六历史系数只用于100ms历史的一次初始化；其field/rk4转交base，不在连续向量场中反复注入历史。对任意有限合法历史和固定有限参数，初始化给出有限初态，不增加状态增长阶数。

因此，在上述固定模型前提下，存在有限常数A、B使`||f(x)|| <= A + B||x||`。向量场由光滑有限多项式/三角函数及常数线性变换构成，局部Lipschitz；Gronwall界排除精确连续解在有限时间内无穷增长。常数依赖每个固定模型参数与图/尺度，不是对所有训练参数统一的数值界。

该结论不保证RK4在既定步长下的粗细差合格、不保证float64不溢出、不保证优化成功，不提供全时有界、稳定域、吸引域、能量正定或L2稳定证书。原全部16事件、完整511点、1e-3数值门槛仍需真实通过。

## 科学范围与最终实现静态核对

六候选各512、三seed共9216更新，保存/排名32、128、512但不提前淘汰，是固定的完整比较；它与新受控语法同时变化，不能把相对旧轮结果的差异单独归因于其中一项。新6201/6202/6203及独立命名空间、池后固定抽teacher、全18端点屏障后一次新合成val，均可保持既定隔离。旧失败teacher、失败seed和6102误选不因本轮重命名而改判。

**最终结论：PASS，可冻结本轮实现并按原登记顺序执行syntax-freeze、唯一相关QA及条件正式比较。** 本次是独立只读静态挑战，没有运行池生成、测试、GPU、数据生成、训练、评价或checkpoint反序列化；此PASS不替代实际freeze、14项QA或任何数值资格，也不是模型采用或L2结论。

- 新模块`polynomial`/`centered_edge_polynomial`/`eligible`（26–81行）用Fraction精确展开、保留抵消，拒绝pi/pj和非多项式算子，并在端点对称与实际机制中心化后再次检查q次数。`generate_pool`（84–96行）直接使用生产池、机制签名去重及固定SHA排序，没有手填候选。`freeze`（99–114行）不足六项返回非通过；`read_pool`与`teacher`（117–131行）将全部池内容及唯一teacher与相同生产规则严格再生比较。该静态核对没有预先获知本轮六项、teacher索引或任何科学成绩。
- `select_seed`（157–204行）先让六候选逐个完成32/128/512，循环内不以排名改变候选列表；真实checkpoint后在32/128重建并调用原共享restore继续。每seed3072、总9216，18最终端点、72检查点、36内部恢复、54排名的计数对应实际循环。共同seed、小批次流、骨架和零闭合初始化保留；teacher/val不在selector参数中。
- root在任何池、teacher或正式数据输出前固定的32/128规则可接受：中间排名只作诊断，不决定淘汰或选择；其数值无效记录保留未决而不单独关闭最终val，与完整512比较一致。`_validate_row`（207–246行）仍核全部0/32/128/512实际count、Adam、RNG、身份、字节与完整trace前后接续；只在512要求排名数值有效，且对全部18候选逐项要求。它没有把旧早淘汰轮的失败改判为通过。模块开头“数值失败阻止val”的概述应按协议第17行及本段的最终端点范围理解。
- `require_all_candidate_layouts`及`validate_selection`（249–274行）先核六候选×三seed完整目录，再读实际状态，检查9216/72/54实际证据。运行中失效候选和异常记录保留，缺任一最终端点使`all_eighteen_frozen=False`。`evaluate_study`（325行起）创建val目录前与结束后重复核冻结证据，独立stream1且只评价train选出的三模型，val不进入选择、调参或追加训练；所有16事件与粗细步长资格沿用旧实现。
- 已发现并关闭一处实际评价接口阻塞：旧`LocalMemoryROM.complexity_counts`（348–363行）使用直接NumPy转换，不能接收CUDA张量。新`load_for_evaluation`（313–322行）现在先CPU加载与复杂度统计，再将同模型移至CUDA。9922测试在全部恢复状态/输出精确比较完成后实际调用该评价加载路径（182–187行），不增加更新、不影响前面的RNG对照；静态顺序正确。
- 私有`launch.py`的REQUIRED（9–16行）与`make_binding.py`的extras显式包含新模块、新tests、复用的finite入口与Tiny helper测试来源，原69项Python来源仍要求与旧实际QA相同。旧QA只用于不变的旧科学闭包，不冒充新源QA。新QA要求同commit、相同全部绑定、typed0、来源前后一致、Job0、GONE及无孤儿清理；通过后才正式run。沿用原Job/runtime选择，不加进程额度、不重建进程监督方案。
- 已关闭freeze证据连接缺口：`launch.check`（56–81行）核实际freeze native/complete及不变来源，并将complete中两份输出的解析路径固定为`BASE/frozen_pool/{pool,teacher}.json`，SHA与正式binding及当前字节逐项相同（71–77行）。正式模块还严格再生池与teacher。未要求通用Job回执中不存在的`complete_sha256`字段。各阶段目录排他、已存在目录/锁先拒绝，未增加覆盖旧结果或外部resume路径。
- 新tests静态共有14case（含五项参数化拒绝），资格更新是9922真实32→34接续36次CUDA更新及Tiny状态2次CPU更新，共38次；正式9216另计。测试覆盖精确次数、生产feature值/梯度、完整比较计数、真实状态负例、全16事件、缺任一最终目录和恢复精确性。测试中的空目录明确只能测试前置布局拒绝，不能充当真实512证据。复用helper不会收集其来源模块整套旧QA；本轮没有实际收集或运行验证。

本审查支持在上述固定受控子语法下进行一次有限算法研究。三seed结构选择即使一致，也只支持协议界定的该子语法/该teacher结果；不能晋升为真实器件机理发现、唯一潜坐标、模型族证伪或全局稳定证书。实际freeze/QA/正式比较若失败，保留原证据并按登记关闭，不依据成绩改变筛选、teacher、512机会或最终数值门槛。
