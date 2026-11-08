# 语法外因果自主实现：独立静态挑战

结论：有条件接受进入本轮唯一 QA。teacher、旧通用学习器、一次初始化和实际32步恢复的实现方向与协议一致。首次 QA 前须落实下列费用覆盖及冻结屏障修正；root 已在准备这些修正。本报告审查的是读取时的实现，不声称已重读最终补丁或已通过任何数值检查。

## 必须落实的具体修正

1. `cost` 的 g_train 初版只计 fixture，而正式 run 另做 teacher 的完整数值资格检查。g_train 应含该检查；g_val 用同一 train 尺度检查。完整评价又核一次 train/val，这是正式工作，应继续保留并由 v_complete 计费，不用移除检查来压时间。
2. `validate_selection` 初版只核目录名和文件SHA，不能识别放在 checkpoint_00512 目录中的真实32步文件。须在任何正式val生成前核真实 training.pt 的 completed_updates=512、trace长度512、固定来源/seed/train_raw/TERM身份、非空Adam状态，以及全三seed行状态；评价后再核一次。费用阶段仍诚实核真实step32。root拟将两次完整文件/训练元数据核验并入每个最终加载的计时，科学计数保持3。增加匹配SHA但真实32步的伪512负例，确认在调用fixture前拒绝；这不授权新测试轮或失败后修复。
3. 正式 evaluate 从train.npz加载 coarse/fine/scales 并copy；初版 cost 的 load 只测模型from_export.cuda，raw_write只写盘和SHA。应在同一load计时实际完成train.npz解包/拷贝；只增加SHA检查不能替代这项工作。数据量和读取次数不变，不需新增计时阶段。
4. 正式每步除 update 数值操作外，还更新总梯度/裁剪序列、首末/最大逐参数范数摘要并检查尺寸；初版cost的u只计update，没有这段逐步记账。body_unassigned覆盖的是本次96次试算主体，按两个进程加入不能直接作为1536次记账的计费证明。将正式同样的逐步记账纳入共用计时更新过程或显式并入u_max即可；保持loss、采样、RNG和Adam顺序。

这四项是完整费用和防止提前val的具体缺口，不要求扩大研究问题、增加候选或追加数值试验。源码须在唯一QA前全部冻结，之后按原失败即关闭规则执行。

## 已核实的实现关系

`RationalTeacher.field` 每次从当前状态取q计算0.4*q/(1+abs(q))；四个RK4子步各重新调用该函数。外部端口加在相同动量方向，再由旧field做惯性换算。生成器从同一初态分别生成dt0.01/0.005，只返回前30维及train尺度；coarse为正式观测，fine为数值参照。正式stream0在run生成、stream1仅在evaluate统一冻结检查后生成。cost使用独立991数据流。没有发现teacher隐藏状态或RHS进入学习器的路径。

学习器通过旧 runtime_model 构造固定tanh(2q) generic_port，加原GRU与六个历史系数；预测沿用旧一次encode自主轨迹。32步时重建模型/优化器后恢复真实checkpoint，恢复NumPy/CPU/CUDA RNG，再进行剩余480步。没有在科学训练中引入teacher q监督、专用饱和或多项式运行类。

当前QA定义覆盖端口零/负q与惯性方向、四个子步、完整0.5 s CPU/CUDA梯度/真实clip/Adam参数、NumPy独立导出、一次history初始化、完整16事件和history数值资格、真实32→34模型/Adam/随机状态/输出精确恢复。CPU端口测试同步修改两层mass，避免上一轮不一致配置。测试定义存在不等于实际通过。

费用原始面板保留96个完整更新（含首次Adam分配），12个真实step32非空Adam保存、3恢复、6构造、3最终加载，三个完整train+val双步长评价；原始写盘含SHA，JSON有固定序列化包络。按各类别max乘正式次数可保守覆盖不同seed；全部面板、外部实际generation/launch时间凭证及body_unassigned必须完整才计算T。外部Stopwatch凭证尚未发生，不可预称已测。131072字节身份填充、512长度包络和262KiB JSON检查只负责固定大小上界，不能替代上面漏项的实际工作计时。

P0助手将QA/cost限定唯一输出和同源前后SHA，绑定旧96项合格源码及新增模块依赖，cost要求实际同源QA PASS。native退出、完整回执、PID/整数FILETIME/GONE用于接受与锁清理。现有助手没有P1执行权；未来P1需另核实际cost准入及三seed统一冻结来源，不能仅凭core的cap=900检查自授权。未发现需要重启旧研究路线的依赖。

## 操作范围

只读新增core/tests、两个P0助手、协议和必要旧函数定义。未运行Torch、导入新模块、测试、推理、数据/权重读取、Git或递归代理；未修改代码。仅写本报告，工具精确wall秒数由完成消息给出，计既有新P2review250。
