# 代理执行合同

角色是职责，不是每轮都要启动的固定人数。root 保留最终科学权；Coordinator 负责执行闭环。
实际模型绑定沿用用户设置：scientist gpt-6-astra，执行 gpt-6.1-sol，有限事实工作可用 Luna。
必须保存宿主实际返回的 model/配置/调用身份，不能用提示词自称某模型。

## 1. 最小任务合同

每个任务有 task_id、role、question、inputs、outputs、write_paths、depends_on、
resources、status、acceptance、stop_condition。可选 priority 是显式排序提示，
不是程序推断的信息价值。输入绑定 commit/hash 或冻结快照；外部证据记录访问日期和出处。
每个任务要说明一句话：此输出帮助 root 作出哪个决定。

同路径及祖先/后代路径只能有一个 writer。复杂写任务在独立 worktree，Coordinator 单点整合。
Reviewer 输入必须冻结；“只读”不能保证读到的正在变化代码是可复现候选。
任务交接给结论、精确证据、未决点、允许修改范围和验收条件，不转发整库日志。

## 2. 按阶段并行

| 阶段 | 可并行 | 必须先后 |
|---|---|---|
| 架构设计 | 数学/可辨识性推导、最近邻文献核查、已有结果抽取 | Scientist 综合 → 冻结简报 → 独立挑战 → root 选择 |
| 实现 | 独立文件模块；只读资格设计 | 模型接口冻结 → 实现 → 复核 → 冻结执行源 |
| 训练 | GPU Job 与不争同资源的理论/报告工作 | 同设备容量不足的重作业排队；不复制相同诊断 |
| 判读 | 曲线/事件事实提取、独立反例检查 | 冻结证据 → 科学综合 → 三决定 → 归档 |

若宿主有 4 个 agent slots，可常用 root/协调 1、实现或数学 1、证据/文献 1、
独立审查 1；按阶段轮换，不要求 4 个都忙。宿主上限变化时以实际容量为准。
单张小显存 GPU 可从一个重作业起步，依据实际显存/RAM 画像决定并发；这是资源互斥，
不是恢复已取消的进程数量/时长额度。模型思考不与 GPU 并发数混为一谈。
高成本 Scientist 用于架构/新 Run 规划/关键科学分歧，不用于轮询、抄日志、机械格式整理。

## 3. 宿主执行的最小事件协议

建议沿用现有 native 执行器：为每个真实 action/job 绑定唯一 ID、已冻结计划、source hash、
允许命令和产物；启动前持久登记，完成后写回 receipt。无法确定是否已执行的动作先对账，
不盲目重放。收尾核验真实进程/锁/归档后才宣布完成。
状态写入应原子替换；活动状态单一写者；失败记录未知项，不能由 LLM 猜作业成功。

本包没有模型 dispatch、后台守护、GPU 执行器、自动 Git 或通用重试平台。
schedule 只输出依赖/资源可行的任务集合，不构成授权；PAUSED_BY_USER 封装输入返回空 ready。
仅给 tasks 列表的调用也是规划查询，宿主仍须检查真实授权与状态。
render-state 仅显示结构化输入；snapshot.live_verified=false 不能用来宣布当前无运行作业。

## 4. 可以借用参考仓库的部分

neural-symbolic-power-rom 的科学/执行角色分离、真实模型回执、action ID、完成事件、
单点状态写入、归档后继续，可用于之后的 host 适配。
不能照搬 TASK4 的数据、目标、state/research.json、runner.py、硬件额度或发布分支。
现有只读 LLM 咨询与预绑定命令执行不等于自动代码生成/整合已完整打通。
先在一个已授权且有意义的小 Run 验证宿主接线，再考虑扩张；不要先建设通用平台。

## 5. 工具命令与边界

    python3 tools/research_harness.py check --root .
    python3 tools/research_harness.py assess --record harness/examples/run_single_negative.json
    python3 tools/research_harness.py schedule --tasks harness/examples/tasks.json --resources '{"gpu:0":1}' --slots 4
    python3 tools/research_harness.py render-state --registry harness/examples/registry.snapshot.json

默认只读；render-state 只有显式 --output 才写文件。不要把示例状态输出覆盖现有 current_state。
证据检查结果是形式记录审查；工具不验证真实数学、不计算统计效应、不证明任务效用最优。

assess 的 exit 0 / valid=true 只表示 schema 合格；必须检查 proposed_*_eligible 和 issues。
不得用退出码自动接受结论。synthetic 的 actual_use_eligible=false；普通记录为 null，
表示本工具没有评估或授予真实采用权限。
