# 正式评价汇总脚本静态复核

**通过本次有界静态复核，可在正式评价完成后执行一次。** 未发现会拒绝既定有效 schema、改变原分母、将诊断子集冒充官方成绩或错误处理 C1 回退的实质问题。绑定脚本 `experiments/osgkd_fit297_20261113/analyze.py` SHA256 `7d67e4dcb7af93a00e78984ccc1bd0f1c6ba56f95f16baf4f519577bc34296bd`，已核对本地字节身份。

本次只读脚本及冻结 runner／评价器／协议，不运行分析、ODE 或模型，不读取任何 npz／原始私有数组；未对正在训练的模型作最终性能判断。

- **断言与 schema。** 读取的 trial 字段、attempts.jsonl 路径及 attempt／event_ids／horizon_s／accepted／clipped 字段均对应 runner 输出。按固定 train_C0、train_C1、eval01 job 名读取，适用于本次登记执行；这不是任意 job 名通用工具。两训练与评价要求 completed、正确 phase、相同 audit_ref、完整 source_hashes、Torch、设备、线程及 qualification_ref。训练摘要须等于 evaluate_release 已绑定的对应 train_refs。
- **分母与官方空值。** 验证全297唯一身份、267稳定／30失稳和原顺序；train 全体及内部两侧、稳定四时域和失稳 actual 均按评价器 schema 解析。covered、unresolved、short_record 列表逐项与 rows 对应；有未决或空总体时官方值必须为 null。输出删除 covered_only_diagnostic 字段，保留 official 和覆盖信息，不以成功子集替代正式成绩。脚本不重算风险及 RMSE 数值，其数值来源是已完成评价回执，不能称独立指标重算。
- **分类。** class_ids 与冻结 summarize 的分支顺序和归类一致：先处理数值／时延未覆盖，再记录自主状态，然后依据 truth_supported 给覆盖内检出／漏检／误报或未知尾段未覆盖。actual 限失稳，8s 含全体；stable／unstable 分母断言正确。差异以同一 split、同一 scope、同一分类列表的事件集合差表示，名称明确 C1_dynamic；多个职责列表可能重叠，不应把输出计数总和解释为互斥人数。
- **共同可比时机。** 使用两模型均有分类、truth_supported 且 timing_comparable 的交集；actual 再限定失稳。核对真值记录时刻相同，并核对有限的 predicted−true 与已保存误差一致，然后给同一事件集合的平均有符号误差和平均绝对误差。无共同事件输出 null 和原因，不补0。数值或时延未覆盖的分类不会产生可比触发时间。8s 未额外按标签筛选，与冻结评价器“双方触发均在真实区间”的比较条件一致。
- **身份与训练配对。** verify 对引用回执核对仓库边界、SHA256 与长度；所有读取 JSON／日志记录实际 bytes/hash 入 sources，脚本自身也留 hash。源采用和物化身份沿冻结 evaluate 回执引用，配对比较逐 attempt 的有序 event_ids 与 horizon_s；只比较两模型共有尝试，超出部分单列，符合两模型独立适配或停止规则。该脚本记录 attempts／rows 的读取身份，不把这些未预绑定文件称作事前冻结证据；其内容与 trial 计数、选点及评价汇总交叉核对。
- **C1 回退。** C0 与 C1_dynamic 的选择分别等于已绑定训练 trial.selected 及 selected_model_refs；C1_score_source 必须符合回退标志，最终 C1 selection 必须等于该来源。成绩、差异和时机仍明确列动态 C1，未把动态成绩误贴成回退后的 C1。回退标志和最优选点由冻结 runner 产生，脚本检查其引用一致性，不重新运行选点；最终报告应按 C1_score_source 引用正式 C1 成绩，同时保留动态模型的登记结果。

脚本仅生成新 analysis.json，存在则拒绝，输出不覆盖；读取范围只含公开 JSON／日志，不打开 checkpoint 或数组。assert 校验要求按正常 Python 模式运行，不使用会移除 assert 的 `-O`。这属于已写校验的运行条件，不增加科学资格门。

未执行真实分析或测试；本复核确认静态汇总逻辑，最终数值和事件数待正式冻结评价产物核对。
