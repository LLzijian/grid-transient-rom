# 训练窗口与更新机会：独立静态审查

**PASS，可冻结本轮四文件实现与协议，并在登记的一次相关QA真实通过后执行固定12模型比较。** 未发现影响本轮数据隔离、实际更新/恢复或评价解释的静态阻塞。此结论为源码审查，不是8项QA、72次CUDA资格更新、24576次正式更新或数值资格已经通过。root负责实际执行准入与最终科学决定。

本审查代理未参与本轮方案或实现。实际读取Astra6采纳记录、最终协议、四个新文件及必要的旧make、prediction/loss、共享checkpoint/restore、实际状态验证源码；没有运行Python、测试、数据生成、训练、预测、评价、torch.load、Git或子代理。只写本报告，不更新旧source-bound审查或实现。

## 科学范围和数据

本轮是根据历史结果固定的两机制train开发比较：沿用teacher2与历史竞争机制3、三份原train数据，分别使用新优化seed6301/6302/6303；两窗口各512/2048端点用于预先登记的配对。选择历史竞争机制、知道teacher身份、复用旧train均已明确，不能将结果称为独立确认、六机制恢复或未见数据上的泛化结论。新轮不生成teacher、pool、轨迹或合成val，也不打开原项目val/test。

`checked_inputs`核固定旧selection SHA与来源、完整旧native/complete及pool/teacher，逐一核三份train数组SHA/位置和原teacher资格记录；原数组为16×511×30，scales为30维、原字节保留，不重算尺度或teacher轨迹。`launch.check`再将旧native的全部来源逐项与当前字节和本次binding核对，因此复用的make、serializer、validator及QA helper依赖仍被绑定；新阶段的源身份与旧科学源身份分开记录。

`fit_slice`只接受50/200未来步，分别取前61/211点，拒绝40维数组和其他窗口。旧`science.make`明确只向`CausalHistoryResidual.from_train`传前11×30历史；后者严格检查16×11×30并从这段历史生成初始化尺度。实际`prediction`只编码前11点一次，将最后一个观测状态与编码q拼接后自主积分；未来观测只在`loss`中作对应窗口监督，不进入encode，也不使用真实隐藏q或teacher系数。

## 初始化、计数和接续

每seed四模型调用同一原make；`runtime_model`复制同一骨架，按相同seed重置GRU/head并将闭合系数清零，历史系数也从零开始，候选2/3均只有一个闭合系数，参数布局可对应。每模型使用`SeedSequence([seed,7])`初始化独立拟合RNG；原train_update每次用同一规则取16事件中的12个，所以在相同更新编号上四模型的事件索引流配对。窗口改变不改变scales、Adam分组、clip100、dtype或历史接口。

固定循环是三seed×两候选×两窗口，共12模型，每模型2048，共24576正式更新；0/512/2048共36检查点。512端点后重建模型/Adam并真实调用原restore，再继续1536更新，总12次恢复；2048只保存和核验，不调用旧restore的≤512范围。checkpoint实际保存step、完整trace、Adam及NumPy/CPU/CUDA RNG，不从目录名推断更新。

最终`validate_selection`逐模型核0/512/2048严格目录、实际整数count/Adam step、来源身份、checkpoint字节、完整trace前后接续和最终latest，合计24576/36/12。共享`validate_checkpoint`/`validate_state`接受实际2048，不含restore的旧范围限制。无外部resume或选择性接续接口；训练失败保留attempted/actual、trace及异常并继续其余登记模型，不覆盖已存在目录或检查点。

短/长组优化积分计数分别为614400/2457600，口径是**批量rollout的RK4时间步**，不乘batch12、不乘每步四次RHS，也不含评价或QA。相同更新数并非相同计算量。协议、counts注释及结果的`optimization_step_convention`已一致；实际wall/CPU仍需另行记录，不能由步数推算。

## 评价、数值资格和配对解释

每模型在512和2048各调用一次`model_metrics`，它只各做一次dt .01/.005的完整train预测，然后由同一对粗细数组计算(0,1]、(0,5]、(2,5]及(0.5,2]四窗口；没有为各窗口重新初始化或分别训练。原score按固定train尺度计算归一化MSE、逐事件/弱强/最差及数值误差界，保留全部16事件与原1e-3粗细差门槛。完整5秒预测/真值数组形状不变，短窗口不会借切片隐藏长窗口失败。

配对差明确为候选3减teacher2，误差界为两者B之和；正差超过该和才判TEACHER，负差小于其相反数才判COMPETITOR，重叠为UNSEPARATED。窗口/更新变化的对比再保留两端误差界之和；这些是离散化误差界，不是抽样置信区间。模块报告长窗口2048三seed是否均分开选teacher及全部预登记差值，不自行采用模型；协议要求的相对短窗口一致改善还须root结合相应contrasts判断，不能只看一个布尔字段。

训练状态冻结与数值评价分开是可接受的：`all_twelve_frozen`及freeze_validation表示真实状态/计数完整，不代表全部metrics合格。`freeze_scope`和协议已写明该范围；无效score或预测异常保留为无效metrics，`paired_profile`对应行明确UNRESOLVED，不能支持该配对科学结论。本轮没有随后val或模型采用准入，不需要因一项评价失败而提前淘汰其余固定训练。即使native正常完成，也不能把无效评价重新标为科学PASS。

## 原生来源与QA

新binding排他写入，包含四新文件、Astra6原始计划/采纳、协议和本报告；旧native全部冻结来源、旧完成/selection/pool/teacher、三个train/资格记录均逐项登记SHA。launcher校验实际文件闭包与已选择runtime，沿用既有Job创建身份和监督实现；本轮不重建Job或加进程数量/时长额度。启动前拒已有阶段目录和相关活动锁，worker目录排他创建并等真实owned握手；run在主入口和worker内均要求同source/phase的实际QA typed0、源前后一致、Job0、GONE、无孤儿清理及全部binding来源相符。QA子Python使用既有DETACHED方案，短临时根qhp1存在时保留并拒绝。

静态可数为8个QA case：六个普通case和50/200窗口各一个参数化CUDA恢复case。CUDA case分别使用独立9931/9932数组，真实32更新、连续2更新、恢复后2更新，每个36，共72；Tiny helper另有2次真实CPU更新，不算正式2048。它们检查61/211、一次前11历史编码、MSE分母、实际状态负例、配对符号/误差界、缺模型/未决屏障和恢复参数/Adam/RNG/输出精确性。测试32→34并不冒充已实际执行512→2048；后者由固定正式流程及实际状态核验完成。

冻结前已关闭两处具体问题并澄清一项口径：QA共同初态对照现在读取已冻结旧pool中的实际候选2/3，不再用手写s*qi替代竞争机制；这不生成新pool/teacher，也不增加更新。积分步数的批量口径现已明确。训练状态冻结不等于评价数值有效现由独立字段说明。这些修改在QA及正式数据/训练输出前完成，未改变候选、seed、窗口、更新数、损失或旧科学源码。

## 审查版本和边界

最终实际读取的四文件SHA256：

| 文件 | SHA256 |
|---|---|
|`experiments/horizon_pair_20261023.py`|`361ab0373f55af225b2dc4ae9514be7c9cbaeef7edb9a2bf042d58d15f509318`|
|`tests/test_horizon_pair_20261023.py`|`8510bfd2eafe19bfe46dd7d2e90dae2db516a17e603c430b583913352c6f2013`|
|`private_data/closure_mechanism_20261019/P1/horizon_pair01/launch.py`|`43da76ec624f18db4590176436e468f8aeaf8140a3d69e5fc4a1ddb6ce804549`|
|`private_data/closure_mechanism_20261019/P1/horizon_pair01/make_binding.py`|`bf6482cf9c65b75650abe681ae5e356c9eadd5df2334b0f79879cc94754ab8d7`|

本次没有执行QA、生成binding、读取train数组或反序列化checkpoint，未独立证明CUDA运行时的数值性、512/2048正式接续或实际退出/锁释放。实际运行必须保留全部成功/失败记录和原val隔离；固定2048后按协议关闭，不按成绩追加更新或挑回512。结果只能解释这两个机制在当前三份train上的窗口×机会比较，不能晋升为六机制恢复、未知物理发现、可辨识性、模型族证伪、全局收敛或L2稳定结论。
