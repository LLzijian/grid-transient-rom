# 固定结构预测分歧通过采集设计准入

2026-10-20。首次真实推断完成：六个固定1024端点在366条train事件上全部通过实际窗口双步长核验；361条事件、24个真实故障母线同时满足预登记的D/B条件，超过12事件/3母线资源门槛。root接纳采集设计准入，可进入下一轮新FOM完整方案登记；默认模型不变。这是已有train上两个经联合适配结构的预测差异，不是真实机制、未见条件泛化或模型晋升证据。

实际gpt-6-astra方案、root登记和最终独立静态审查见[协议](first_disagreement_inference_20261020_protocol.md)。固定full8/delete7各6101/6102/6103，不选best seed；393原始分母保留27排除，参与325稳定和41失稳样本。各自仅100 ms历史初始化一次，真实评分N=3–500、不补尾、不按预测失步裁短。CPU float64、fine .005/coarse .01各一次，2196项模型—事件数值核验全部通过。

21维可观测量使用原M和train尺度；B为两结构均值差平方范数/4，W为每结构三个seed的样本变异、再平均两结构，epsilon=1e-6。事件等权的fine D中位数13.32821080977477、范围0.31031774109714594–61.0185465974119；coarse中位数13.328214068611942。fine B事件均值0.0890410798965215，W事件均值0.0460183384890122；均值比不能替代逐事件D中位数。两步长均D>=1且B>=2.5e-5的361/366事件覆盖24母线；其余5条保留，未删后重算全体。

执行源1cc0fc89aec791d3ef80e42d396361714c332342。两层真实退出0、无超时，实际worker87180/内层launcher89904/PowerShell49128/外层82304均退出；FILETIME身份匹配且清理核实。70项源前后一致，当前源无变化，子进程自身结束前复核同一合法模型/数据清单。最终JSON SHA a8fcae5de3c363dc024fc8621c13c9b3af904bf799cf377e1fe5fa7883e4ccd6；12份本地双步长预测文件哈希全核实，未再推理。原始大数组保留private_data/closure_mechanism_20261019/P3/first_disagreement_inference01/run01；完整私有结果/来源及紧凑逐事件证据见[summary](../results/first_disagreement_inference_20261020/summary.json)与[事件表](../results/first_disagreement_inference_20261020/events.json)。重现科学结果需已有私有train和六源检查点，不声称GitHub含全部数据。

唯一真实外层包络122.6443724000128进程秒，包含内层121.3487221，不重复加；固定launcher总122.7177061的有限外层附加开销纳入支持保守900。结项1022.6443724；登记未用3477.3556276明示取消回P3。父累计54013.397739/259200、余205186.602261；P3余91673.9293648，其他类别与旧900支持hold保持。

复现来源和完整命令已在摘要中：.venv-gpu/Scripts/python.exe private_data/closure_mechanism_20261019/P3/first_disagreement_inference01/launch_once.py --source-commit 1cc0fc89aec791d3ef80e42d396361714c332342。该目录已完成、禁止重跑/恢复或覆盖；未来复现需新授权新目录。私有launcher全文/哈希已归档，调用原native_job外层及PR12监督，不另建平台。

独立代理/root/fixed_edge_final_review未参与提案实现，只读核对原生回执、70源一致、最终哈希、393/366分母、2196数值资格以及已存D/B的361事件/24母线计数，未重算D或读取大数组。其意见同意采集设计准入及有限工程PR，主张仅到下一轮FOM设计。root采纳；新协议必须先容纳采集、双步长、六模型统一256更新与冻结后新val确认全部费用，不能直接据本次结果启动FOM。

原P3旧exitnull/整HDF5字节读取违规及拒绝manifest、原6102删除512数值未决均保持，不用本次成功追改。新轮仅合法train，未进行val/test评价，无训练、Jacobian或FOM；原test关闭、TASK4暂停、生成搜索/P4门槛和默认模型不变。合格实现与证据经范围有限PR、预期head及远端核验后方可main集成；模型不采用。

PR13已合并并核实：预期head c34095faa7c8e5cf3decdddf622c04a8cc0de442，远端main e126a01f34c45bdda1d1d0588bba93dd25e0a4fc。11个相关文件，实际资格及独立审查通过，无远端CI任务；不采用模型。
