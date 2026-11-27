# IEEE39 独立确认 v2 最终冻结审查

**Verdict: PASSED。v2 候选可封存为本次唯一待授权主体；当前尚无 test 开启授权，未执行终评。**

本次审查绑定 `results/osgkd_v2_frozen_confirmation_v2_20261128/freeze_candidate.json`，SHA256 `f48aaaf5faa75e5848b1b325318352f85521662fea882901531f248e5e1a079f`，evaluation ID 为 `ieee39-independent-expansion45-supplement41-20261128-v2`。root 可增加独立通过状态和本报告引用以形成正典 freeze；其余内容身份、端点、比较、数据、数值与权限条件须保持本候选一致。此通过不代替用户明确的一次性授权。

## 实际来源与前序闭合

本次观察 Git HEAD 为 `76150bacd157b96dc84d910a5adfecf0665ba4bb`。补种与 v2 材料仍包含未提交文件，AGENTS.md 有既存修改；审查依据实际工作区逐文件内容哈希，不将所有材料冒充该提交的内容。87 项源码/协议来源哈希逐项匹配；候选内全部 84 处 path/bytes/SHA256 引用均核对一致，包括 31 份 checkpoint 的外部字节身份，没有反序列化权重。

WP0、WP1A、WP2、WP3.1 和 WP3.2 补种共五个前序的 registry、独立复核引用与闭合状态匹配。执行为 COMPLETED、科学预算为 closed；WP0 精确绑定既有通过复核，其余通过状态明确。补种 registry SHA256 为 `43a1ecc4c48dc7b8bb7fa38219324d35fe7e0c1d757da3d14ae4573a6d588e24`，最终独立复核 SHA256 为 `612ba9ae374f485f222b7a9ceae76be58414b36a4af468e89ca7390708cefa12`。两个新增扁平末点和一次 val 的训练/原生/评分链已在该报告独立复算，本次不重复评价。

WP3.2 新决定 SHA256 为 `ad1c9dfc902a9353702b5db2f7e9198e575059578c0395d7082c5bcfe59be1aa`：status closed、extra_flat_seeds completed，local_stability_and_modal deferred，并精确引用第五个前序及其独立复核。候选如实记录此前原始 val 暴露，包括补种新增两末点的一次评价；冻结准备本身没有重新装载 val。

## 31 个端点与 52 项比较

31 个端点 ID 和 checkpoint SHA256 均唯一，逐个与登记来源 summary 的 checkpoint 引用比对一致。构成为 8 个第 2048 步端点与 23 个第 128 步端点；v1 的原 23 个端点逐项原样保留。补充的两个 flat 来自已独立闭合的本次补种，no_beta 和 frozen_p 各三个种子来自已封存 WP1A，对象与预算无替换。

`static_24121` 继续作为 `full_24121` 别名，沿用 v1 已审查的相同限制、128 次尝试及 0/64/128 模型/Adam/RNG 一致性证据。别名不新增积分或独立样本。

独立重建全部有序比较并与候选逐项匹配：

| 比较角色 | 对数 |
| --- | ---: |
| primary | 1 |
| matched_open_model | 1 |
| matched_encoder | 3 |
| paired_refit | 18 |
| registered_parent | 23 |
| matched_static_mask | 6 |
| 合计 | 52 |

`full_24121→top30_24121` 与 `full_24121→top20_24121` 各自担任 paired_refit 和 matched_static_mask 两个角色。源码使用 role、reference、candidate 共同组成公开和私有配对文件名，52 个名称均唯一；两个角色分开留档，实际仍为同一对模型，不当作两份独立证据。其余配对 ID/方向/顺序均与协议一致。

去动态调制保留静态互易边；冻结机械动态保留估计机械初态和机械作用，并将相应状态保持常量。限制组对 full 的比较是在同预算再适配下的取舍，对登记父点的差还包含新增训练机会。图/扁平比较覆盖共享方式、容量和联合优化路径，不归因于单一结构因素。

## v1、合法来源与原分母

v1 正典 freeze SHA256 保持 `e4895159fbb7de1769ab6cb27c8a7328258b1dbdd511904ef57976e3a58cb892`，状态 frozen、未授权且未访问 test。v2 的 supersedes 引用、data_identity、constants_source 和 numerics 与该 v1 冻结逐值一致。未改写 v1 文件或历史决定。

重新读取合法 split/index 清单并核四个内容哈希、白名单摘要、事件与物理场景无重复；只 stat H5，不打开其内容。计数如下：

| 来源 | 原始 test | 稳定 | 失稳 | 历史 11 点资格 | 其中失稳 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expansion | 45 | 43 | 2 | 43 | 0 |
| Duration supplement | 41 | 24 | 17 | 33 | 9 |
| 合计 | 86 | 67 | 19 | 76 | 9 |

已消费 Pilot 60 条不在白名单中。三个故障母线组与共同系统、控制器和参考构成相依的事件族，31 个模型和 52 项比较共享这 86 条事件。历史 76/9 是可初始化预期，实际数值覆盖必须在授权后按统一双网格判定；原始 /67、/19 不随输入或数值资格删除。

Expansion H5 stat 为 1324250705 bytes、mtime_ns 1788881772375386300；supplement 为 169957630 bytes、mtime_ns 1789036181743352900，与候选及 v1 一致。历史覆盖回执哈希匹配。此核验不等于 H5 全文件内容认证；首次明确授权读取仍必须核冻结的文件内系统签名、单位、共同参考和白名单。

## 主张、数值与权限

主张台账 SHA256 为 `61a87067b3235fab7f92e92faa2fc5c027ff32039784d2a674f3f8266f71753c`。三种子图/扁平结果与刚闭合的补种独立复核一致；新增消融计数与 WP1A 封存表相符。台账区分稳定风险、误报、自主召回与真实区间预测，保留固定预算和裁剪边界；Layer 1 支撑恢复、公开角色映射迁移和机械摆阵模板重拟合分别取证。它不预先宣称 test 成功、结构唯一性、跨电网部署或安全证书。

方法稿引用对象仍为 `paper/osgkd_v2_methods_and_theory.md`，SHA256 `9e119ac8f75e4cef1e1c9384442c2992613000bc1c6f7d2db42089b4c751954b`。本审查不重做原数学证明；其功率、全局存在性与反例结论仍受原假设约束。

部署保持 11 帧、100 ms 因果运动历史与冻结常数，保留共同转速；初始化不输入标签、未来状态或逐事件隐藏运行点。CPU float64、共享 Python、单线程、批量上限 16，0.01 s 输出、0.005/0.0025 s RK4 内步长与原阈值不变。没有失格后换步长、末点选择或权重更新。轨迹误差截断在真实记录内，稳定窗口不完整/数值失格时正式风险为 null；自主 8 s 触发与真实区间检出、时间比较不能混称。

v2 入口必须先验证正典 freeze、第五个前序、独立复核、闭合决定、源哈希与明确 v2 用户授权，再登记共享暴露声明并物理读取。首次读取保存回执和完整私有快照；完整缓存可供同授权/evaluation ID 技术恢复，完整端点复用。已暴露而无完整缓存时拒绝重新打开；科学错误写停止标记，原入口拒绝续算。

跨版本边界明确为：**v1 只作历史，不得在 v2 授权或暴露后再申请/使用 v1 授权。** v2 会拒绝已有 v1 授权/读取回执；未改写的 v1 代码自身并不检查新增共享暴露文件，因此不能声称两个旧入口已自动形成双向互斥。root 必须维持只授权 v2 的操作边界，已发生访问不得因版本改变而重置。

本次检查时，v1/v2 均无 authorized_release.json、test_access_receipt.json、data_complete.json 或正式 summary；共享白名单暴露声明亦不存在。候选明确 user_test_authorized=false、test_trajectory_access=false、official_updates=0、new_fom=0、default none，并保持保护锚点约束。root 后续封存、提交、推送和同步后，仍需等待用户针对 v2 固定清单的明确一次性授权。

## 实际审查回执

Reviewer 为 `/root/wp3_reviewer`，与提案、实现和 root 冻结材料作者分离；宿主服务器解析模型身份 unknown。本次只执行源码、文档、JSON/CSV 与 checkpoint 外部字节身份核对、H5 stat，没有打开 H5/NPZ、反序列化权重、调用原始装载/评分入口、训练或积分。唯一写入为本报告；未创建授权、暴露声明或修改 v1。最终结果待授权执行后再从保存逐事件证据独立复算。
