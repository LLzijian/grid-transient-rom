# Run3 首次 I/O 续行：冻结实现独立审查

结论：**accepted，具备由 root 释放唯一续行 action 的条件。** 此实现保持原 Run 的模型、输入、数值门、选择规则、评分函数和分母；仅继续尚未开始的 553 次初始化及原定后续分析。当前审查的释放文件仍为 `released=false` 草案，尚未授权或执行实际续行。

本次为源码、保存证据和资格回执复核；未调用续行程序、初始化、拟合、积分、谱计算或测试，未读取 `fit_events.npz` 字节。不构成新的独立科学样例或结果证据。

## 冻结引用

以下 SHA256 均已核对当前字节；原 51 项源绑定完整继承，草案 53 项源和 89 项非事件缓存资产均匹配。原 21 项资产全部继承，其中事件缓存按原哈希登记，留待排他 claim 后核字节。

| 文件 | SHA256 |
|---|---|
| `experiments/gftd_fit297_20261107_resume01.py` | `21af1edcbe2c5d5740866b62098e5e1390a496a98132d3101f7349d3210b0ff5` |
| `tests/test_gftd_fit297_resume01.py` | `9bea043b8a75566b859a6b956fc5e0c7b2f544d82d95299075b041a8d1dbcd30` |
| `results/gftd_fit297_20261107/resume01_qualification.json` | `6d4b2ad48c3861c8792780986b3338060869807de9343f6cb2f45e4f267ea1b9` |
| `reports/gftd_fit297_20261107_resume01_design_review.md` | `3f16b5a11a5f7d44110fc2c472a51538a2f03278023efbb94c2ec64e8e21d5a3` |
| `results/gftd_fit297_20261107/resume01_implementation_adoption.json` | `505044e2a6d86b4048f70afa6c1bad19276d76edf36a2d20787355515f2d6c5a` |
| 原 `execution_release.json` | `f10a023faf1783db600230c0ab370cb9f919e73dbeaf08b470a0b9c0bd160443` |
| 原 `execution.partial.json` | `98c570de7f466c2d7397faf021f8433cc7be7ac7070c1bd866c77a5a48f719d9` |
| 原 `execution.partial.json.tmp` | `f32ffb1aace07274ab3b3f38dd0fea6527a1d1ad71533df0c6b8d1ee6f4c20b5` |
| 原 `static_model.json` | `97714cfed13b4e243924d87a6de7590f76bd7fded68c4322d22d4f907fa4c032` |
| 原 `modulated_model.json` | `86adef0b5db08f56594ebbd4b78924e6b9fb43e7aa2c47f1d8fc91baec1ea7a5` |
| `resume01/execution_release.draft.json` | `9872197a7542ab16fff8f98c0e02c12ddce5626d68aebef94dd251c9de67f05b` |

表中相对文件均在 `results/gftd_fit297_20261107/` 下，模型在 `private_data/gftd_fit297_20261107/` 下；显式完整路径除外。原协议 `ca22743c53fe4fe398960e111838a373f8c4686eab0974c5daf6d9c28435062f`、设计、归档、静态身份审计、物理参考和 FOM 来源随原资产绑定继承。

## 实现核验

1. **无重放边界成立。** `restore`（92–143 行）要求恰好 `static_000..040.json`，绑定事件顺序、模型准备键、适配索引和全部旧成本；保存状态必须为有限 float64 50 维且末点 p 逐值相同。独立汇总 41 份诊断，得到 8 个成功状态、49 个优化任务、9347 次残差调用、396020 次 RK4 步，和完整 tmp 逐项一致。33 次 stage1 数值失败保留原失败结果。新顺序固定为 static 索引 41..296，然后 modulated 索引 0..296，共 553 项；首项为 `expansion_0044`。不拟合、不重复前 41 项。

2. **成本和缓存处理正确。** 162–186 行继承完整 tmp 的两次拟合、41 次初始化和耗时；每次新初始化先记真实成本，再写状态和诊断，新增上限为 1106 个任务、995400 次残差调用。分析账仅恢复最终累计的 information SVD 416、条件投影 315、memory SVD 594，不重复累加中间累计账。内存 `_CACHE` 保持空；195–203 行调用原 `coverage_information` 的缓存缺失出口，要求 `unknown` 且新增计数全零，注明中断来源。原 24 包 fit 信息保留；全 train 部署信息覆盖未完成，不能解释为信息不足的实测结论。

3. **原后续科学流程保留。** 191–193 行先冻结全部 594 个模型—事件初始化结果，再加载评分事件。223–282 行与原主体 409–468 行对应，复用原主细未来积分、L1 失败/评分/配对函数、L2 证书及点检查、ROM 模态、既定 Kron、固定 descriptor、全配对模态及图比较；FOM 身份取原 release。原 297/267/30 和 594 对分母不变。原数值失败不会因恢复而重新尝试或从分母消失。

4. **绑定与读取顺序符合恢复合同。** 58–89 行检查独立 action、已终止父进程、原环境、设计和采纳哈希、原源及资产继承；151–152 行排他创建新 `started.json`，重复 claim 拒绝。事件缓存在预检查阶段不读字节，206–209 行在 claim 和全部初始化冻结之后按原哈希核验，再调用原 loader。草案额外绑定资格、快照 manifest、旧 host、41 份诊断和 8 份状态；最终 release 需按 root 原计划再绑定本报告，主机显式沿用草案三个单线程环境变量。

5. **I/O 修复不增加科学机会。** 33–45 行为每次写入生成唯一临时文件，flush/fsync 后仅对同一已计算字节的 `os.replace` 做最多 100 次 PermissionError 重试，间隔 0.05 秒；不重跑数值步骤。终止失败保留待替换字节。284–317 行保留首个异常和 traceback，若最终回执再次失败则另写唯一 emergency 文件，并在原异常附注第二次失败。旧 partial/tmp、源和资产不覆盖；结束时重新核绑定。

6. **耗时口径明确。** `resume_wall_seconds` 是 claim 后续行主体时长，原程序完整 wall 为 null；父 host 单列，`combined_host_seconds=null`。累计模型部署成本为原已保存初始化耗时加续行初始化/积分耗时，不冒充一次连续主机运行、CPU 时间或 FOM 加速比。

资格回执记录最终 21 项工程测试通过、14.86 秒；前两轮 fixture 缺键导致的失败及修订仍保留。测试覆盖精确 553 项顺序、全结果先冻后评分、旧账、原辅助缓存未知出口、超预算拒绝、重复 claim、同字节重试和双重 I/O 失败。资格支持恢复编排，不证明真实模型拟合充分或科学效应。

代码决定：接受该有界技术续行实现。模型决定：原拟合模型及保护指针不变，当前不晋升。科学决定：允许 root 释放原 Run 尚未执行的工作；完整结果仍待结果审查，缓存丢失造成的辅助 coverage unknown 保留。未发现要求修改冻结源或增加科学前置条件的阻断项。
