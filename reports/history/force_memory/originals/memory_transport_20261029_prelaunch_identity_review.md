# 启动前保留锁身份：独立工程挑战

2026-10-29。实际模型 `gpt-6.1-sol`，独立只读角色。仅核指定保留锁/恢复记录、原身份函数及两个目标身份；未启动Job、数学资格、性能、评价、NNLS或训练，未修改源码、锁、状态或预算。本公开报告不包含实际PID、FILETIME或绝对私有路径。root保留最小修订、归档重冻源及一次启动重试的决定权。

**条件结论：足以支持对这一个精确历史锁的最小准入修订，接受GONE或REUSED_PID证明记录中的旧owner已不存在。** 不支持全局放宽锁规则，不授权操作复用PID的新进程，也不证明旧Job或全部子树当前已获实际清理认证。

## 独立核得的事实

- 保留锁SHA256为 `f9d72f0d5fccd8b5349907d69299aad543aa9b936f3e016971a7e24acd57c3e4`；恢复记录SHA256为 `e53d9b82c8a19d5a3899cbd04682a298f16874bce27215e18d2b297e654e709a`，均与root指定值一致。
- 恢复记录status为INTERRUPTED_RUNTIME、lock_preserved=true、指向同一lock SHA。supervisor/worker均通过原typed_identity的正整数PID与完整creation FILETIME检查；锁中两身份逐项匹配原冻结常量，恢复记录同一两身份的历史state均为GONE。
- 使用原 `identity_state` 对这两个身份作一次只读核对：supervisor=GONE，worker=REUSED_PID；没有读取或打印当前复用者的其他信息，也没有向其发送操作。
- 新阶段base存在，但prepare结果、native目录、ownership目录、optimizer账本、qualification目录及active lock均不存在，符合失败在prepare launch前发生。root报告首run实际exit1、未产生任何native或科学工作；本审查没有重跑失败命令。元数据base存在不代表已执行prepare。

## 原理与最小必要边界

`uncapped_windows_job_20261021.py:58–69` 先以查询/同步权限打开目标PID，再比较完整creation FILETIME；只有不等于旧值才返回REUSED_PID。Windows活进程的PID唯一，所以它证明当前PID属于另一进程身份，记录中的旧owner已终止。该函数仅查询、零时等待及关闭自己的句柄，不kill、不分配Job，也不需要等待复用者结束。helper实际SHA256仍为 `551560fc332497c5f5580cac91c85877c444ebbf0f1886238b4110bf9a361cb3`。

原 `r_memory_execution_20261025.py:99–120` 对此唯一保留锁已要求路径/lock SHA、恢复INTERRUPTED_RUNTIME/保留状态及两项冻结身份/历史GONE一致，但:117当下仅接受GONE，因而把已不存在的旧身份误拒为复用。最小修订应局限在新loop的这个已知例外：先核上述精确lock/recovery SHA、全部原字段和typed常量，再接受实际GONE或REUSED_PID。RUNNING/LIVE/ALIVE、UNKNOWN、ACCESS_DENIED等返回值及任何查询异常仍硬拒，身份/哈希不匹配也拒绝。

必须分别保存“历史GONE记录”和“当前REUSED_PID”，不得把当前复用映射成伪GONE后仍输出原硬编码状态。可使用已有新阶段的准入证据记录这两个真实状态及哈希，不需要新注册系统。旧锁、原恢复记录、原函数源码、旧binding和失败stderr保持不动；其他锁、当前owner认领、runtime复用及科学数值门继续走原检查。没有对新进程的kill/Job/锁删除授权。

这次例外不新增历史Job成本或清理事实：原函数本就只援引历史owner和中断记录，不应把接受REUSED_PID进一步写成有实际Job accounting、无所有孤儿或整个旧子树已重新核清。需要的结论仅为这两个完整旧身份均已不存在；其余启动前要求必须独立成立。

原启动失败及exit1须保留，修订后重新归档/核源并由root明确决定一次prelaunch retry。1848科学调用账不重置，CPU/CUDA/源平面旧资格不重跑；本报告不构成真实loop释放，也不预判后续性能、QA或研究结果。
