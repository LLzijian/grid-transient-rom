# OSGKD 最终复核（当前完成 C0／C1 训练部分）

**C0 已完成原登记训练与一次适配，训练回执及日志复核通过；当前尚不裁决 C1、正式预测评价或最终科学结论。** 本次只读取冻结 JSON／JSONL 和公开源码，计算日志计数及已有标量关系，并核对引用文件 hash；未执行模型、ODE，未读取 npz 或 checkpoint 权重。

## 输入身份

| 输入 | SHA256 |
|---|---|
| `results/osgkd_fit297_20261113/train_C0/summary.json` | `6631684fd1ca446cd4c6876265b9e0069af72b357bd5e1b7f1c07ef9ca51996e` |
| `artifacts/osgkd_fit297_20261113/train_C0/C0/attempts.jsonl` | `f4ca8729446a1741d61c396a304a93f9fb1b18af9a6085f44f41307f0bdb1e9f` |

回执为本 Run 的 train_C0、phase=train、status=completed、variant=C0，参数数6809。release 记录单线程 CPU、float64 对应执行、Torch 2.13.0+cu130。逐项核对 release.source_hashes 与当前公开文件一致；audit、qualification、implementation_review、data_review 和保存的 release 引用 hash／字节数均一致。资格回执 completed 且 passed=true，source_hashes 与 audit_ref 与训练 release 完全相同；观测发布状态为 adopted_for_prediction。未重新读取私有 bundle 验证数据。

## 执行与选点

- 日志实际1500行，attempt 按1至1500严格连续；逐条 accepted=true，汇总 accepted=1500、rejected=0 与日志一致。预热0.2 s有100次、0.5 s有100次、2 s有1300次，符合1000次基础尝试加500次适配。全部日志事件属于冻结内部拟合 ID；C0/C1 的逐尝试配对尚待 C1 冻结日志。
- 适配确由原第一分支触发：600和1000检查点内部拟合 J 分别0.21990649272721194、0.18539104302518886，相对下降15.6955118851%，超过原1%门；两点数值探针均通过。第801至1000次全部200次接受，其中3次裁剪，无非有限拒绝。因此 fit_J_improved、保持原学习率继续500次，与冻结 runner 逻辑一致，不依赖 C1 输赢。
- 保存的检查点为0、200、400、600、800、1000、1250、1500；各点内部拟合／检查目标无失败事件，数值探针均通过。按正式可选集合排除200，并按最低内部检查 J、相同则更早 attempt 独立检查，最优确为1500，check J=0.24125890267137132；selected 对象与该检查点完整一致。没有从最终预测评价反选的证据。
- 日志裁剪28次，占1500次的1.8667%；裁剪前梯度范数范围0.09691813615095458至20.044776882299423。逐条 J 和 gradient_norm 均非空且有限。裁剪是按原 clip=5执行的优化记录，不等于候选动力学数值失败，也不证明优化已经充分。

## 当前裁决和待补

代码／执行证据：C0 本次训练记录与原资格、输入和有限适配规则一致。模型：C0 已产生训练选点，可按原约定进入一次正式评价；这不是模型晋升。科学：当前确认完成本预算下训练，不由低裁剪、全部接受、达到1500次或内部检查 J 推出泛化、机制辨识或优化充分性。

C1 训练回执、两模型配对、正式评价主细资格、全分母指标、分类与时机、C1 回退及最终报告结论均待后续冻结后补充。未读取权重或重算轨迹；该部分不替代正式评价证据。

## C1 冻结训练回执补充复核

**C1 训练回执和完整1500次配对序列复核通过。** 以下补充更新上节待办：C1 训练与配对已完成复核；正式评价与最终报告仍待完成，不读取正在生成的评价产物。

| 输入 | SHA256 |
|---|---|
| `results/osgkd_fit297_20261113/train_C1/summary.json` | `77c2ddd21d33adeccfed98e22a93f39efa4a8f408be167f01c29670364ce5ca0` |
| `artifacts/osgkd_fit297_20261113/train_C1/C1/attempts.jsonl` | `558a136407275f21b496e04f3ff252d05a6fbdac9d466a1aa40a5390ca274721` |
| `results/osgkd_fit297_20261113/evaluate_release.json` | `d72483e182bc51cf1c20b18eb2eb6de2b5e380f7258158b97afb4bcd3a48b635` |

C1 为 completed 的 train 回执，参数数6854。日志严格为1至1500，实际逐条接受1500次，0拒绝，裁剪22次（1.4667%），与摘要一致；全部 J 和裁剪前 gradient_norm 有限且非空，梯度范数范围0.07331597821787553至18.961217932361667。时域计数仍为100次0.2 s、100次0.5 s、1300次2 s，stop=budget_complete。

适配依据为600次 fit J=0.21725373126967798、1000次 fit J=0.18224890455113862，相对下降16.1124168105%；第801至1000次全部接受，其中3次裁剪，两基准点数值探针均通过。因此原 fit_J_improved 分支成立，追加500次且保持原学习率。八个检查点均无内部拟合／检查失败事件且 numeric 通过；独立按正式集合最低内部检查 J 选点得到1500，J=0.2399448060318834，selected 与该检查点完整一致。

全部C1日志事件位于冻结内部拟合清单；两模型整个1500次尝试的**有序事件ID序列和horizon_s逐条相同**，没有只核对共有子集后遗漏剩余尝试。两训练 release 的输入引用及source_hashes一致；C1当前公开source hash、audit／qualification及保存release引用hash和字节数核对通过，资格仍为completed／passed。

对evaluate_release仅作静态核查：phase=evaluate、job_id=eval01、variant=null、CPU单线程，输入与两训练一致，train_refs绑定各自完整冻结摘要，两份selected_model_refs对应已选1500次artifact引用。qualification_ref的JSON键排列不同但路径／hash／bytes相同，不构成身份差异。由于动态C1内部检查J较低，按冻结choose_pair规则不使用静态回退；这是已冻结选择规则的结果，不能解释成正式预测性能结论。

当前模型决定为保留两训练候选进入既定正式评价，不作晋升。低拒绝、低裁剪和内部检查J均不等同优化充分或独立确认；正式评价的覆盖、数值门、误差与分类尚待冻结结果。未读取npz／权重、未运行模型或ODE。
