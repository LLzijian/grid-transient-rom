# 当前研究状态

任务RUNNING，cwd E:/03-nn_Transfer_Func/kiki。新检查点续接GPU作业正在实际运行：会话22995，worker32344/FILETIME134371383121302454；源6bf8f730e4fc180af9e815682a8fb4fe4b7fd020，原生Job拥有rational_checkpoint02/active.lock。禁止重复队列或修改绑定源。最近轻量观察确认worker仍RUNNING，0_rational_6301完成460步（其中旧32、新428），当前尚无完整组512冻结记录。该数字仅为一次进度观察，非最终结果。已通过一次相关CPU测试12项，以及唯一四路径132次CUDA更新的逐值资格核验；三份旧32状态全部迁移成功，现正式续完6048新增更新。旧96单列，正式总数仍6144，不能把资格132计入正式进度。

资格证据：[CPU摘要](../results/rational_execution_20261021/checkpoint_unit_summary.json)、[CUDA摘要](../results/rational_execution_20261021/checkpoint_cuda_summary.json)。两形式连续0→33与0→32真保存/重建/共享恢复→33，模型/Adam/NumPy/CPU/CUDA随机状态/loss/梯度全部逐值相同；迁移6301的32状态也与新资格轨迹逐值相同。三个旧32各加载一次、保留原identity，发布新source/来源32后计算状态不变；同一进程保留活模型继续各480，其余九模型各512。配置沿用旧真实训练仅线程数1，不借probe的确定性设置追平。

此前旧source5ffe63f5f9bd71c68e417da4759fdbd183299799的0_rational三seed各32后诊断命名冲突，3×UNRESOLVED/0×512，原complete、96更新、源和检查点全保留。[失败摘要](../results/rational_execution_20261021/first_gpu_failure_summary.json)。完整旧81文件入口集成已撤回；不QA07、不重跑五例或QA06、不把已闭合组伪装中断恢复。新模块/测试/私有入口新增，旧134绑定文件字节未改。第一次unit开跑前遇到旧PID复用准入拒绝，未开始测试；重新核11个旧身份实际GONE后唯一12项CPU集合通过，没有放宽PID规则或修改旧源。

已合并工程进展：只含Job监督器、运行环境helper、16项隔离runtime测试及范围说明/摘要的六文件PR #30已审查合main，远端3cab35e4bb9bfe29f282d3ea16267f574d7dabef已核expected head94fde44958188494c6f6e0daff787b1c4d0c9ea3/父提交/树/文件。原五例与188项QA06的成功仅用于其实际覆盖的Job/helper范围。[远端验证](../results/rational_execution_20261021/windows_runtime_main_verification.json)。新检查点模块尚未合main，模型采用none、L1/L2未完成，工程合并不等于科研晋升。

下一实际动作：健康作业按3–5分钟/50–100步稀疏检查，四组顺序0_rational→0_tanh→1_rational→1_tanh，均6301/6302/6303；所有12个512端点真实核验冻结后，同一已授权作业才执行一次原合成val。任一未决保留且不打开val。当前新增入口不支持外部--resume；如中断保存关闭后另行登记来源续接，原最多一次非择优界限不重置。内部原定32真实恢复已实现并实测。

原项目val/test关闭，TASK4暂停，40状态/100ms一次因果、真实区间/失稳样本、数值与模型门槛、数据条件配额保持；旧18×5126085.6472446和384测时不重跑。旧原件在rational_execution01，新science/qualification/migration_read在private_data/closure_mechanism_20261019/P1/rational_checkpoint02；实际命令launch.py run --source-commit 6bf8f730e4fc180af9e815682a8fb4fe4b7fd020已经执行，禁止再启动。

全部进程额度取消，实际wall/Job CPU照记。六旧QA wall466.6981428、五例41.5592856、旧首次GPU152.273702、CPU资格32.7407797秒分别记录；当前GPU作业费用待真实Job回执，不造精确进行中费用。未知支持成本/历史hold/REJECT保留，费用账本results/closure_mechanism_20261019/cost_snapshot.json。归档分支codex/edge-memory-discrimination-20261021，资格/迁移证据已核63633c7c85e8c7e53365d71b535e0b7dd7bd54f8，main合并记录已归档b4bba6ceb3121a6038edf607c0a9175fa538945f。local HEADa95101337855ed3fc1f4abfd710982443d5564cb和index SHA82b060840d50257c41df55e15805fa106f29117cac8bf1868a6b83e490e2df93未改；不reset/强推/删证据，heartbeat保持。
