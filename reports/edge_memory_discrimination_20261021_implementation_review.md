# 两种边记忆结构研究：完整实现的独立静态审查

2026-10-21。结论：**完成下面两项必要静态修正后，可以进入预登记的唯一P0 QA；本审查不宣称数值正确性或历史对照合同已经通过。** 没有运行Python、Torch、pytest、生成器或科学作业，没有读取轨迹、合成val、项目val/test或旧checkpoint。审查只读新runner、tests、两个新helper及原训练/fixture/evaluate函数片段。修正和最终源码归档必须在QA前完成；QA或cost开始后仍遵守原单次规则。

## 必要修正

1. 新runner `_worker_receipt`写入`created_filetime`与`creation_filetime`，但launcher成功门槛和锁释放均读取`creation_filetime_utc`。按当前四文件组合，真实QA必然无法满足完整进程身份门槛。须补齐与原native监督器约定一致的整数`creation_filetime_utc`；不能通过去掉FILETIME检查绕过。
2. `_train_fixture`检查coarse有限性，却未检查fine及归一化双步长差的有限性；fine出现NaN时`NaN > 1e-3`为False。须显式核coarse、fine、scales和归一化差均有限，scales为正并保留原下限及1e-3门槛。不能借JSON最终拒绝NaN代替科学数值资格检查，尤其cost输出并不保存全部fixture元数据。

## 已落实的实质条件

- train-only生成器准确复用stream0、16条/60步、coarse与fine、原固定尺度及clean_sha256[0]核对；formal=True才使用stream1/110步并核clean_sha256[1]。没有调用原同时生成train/val的整体fixture。隐藏q只在teacher内部生成；返回仅30维观测。
- QA旧对照核验为零更新，逐源重构共同未拟合source，核stage0全部state_dict、参数顺序/flags、空Adam、NumPy/CPU/CUDA RNG、零闭合系数及六头系数；读旧对象后恢复捕获状态。512端点计数、trace长度和文件哈希进入核验。上述是检查代码存在，不等于已实际读取/通过合同。
- generic参考独立解析AST标量值，对内部完整能量autograd，再组合J/R/B并转换动量；没有调用新增闭合解析梯度或快速field。tests覆盖两结构、非单位质量、完整0.5秒损失、全部原始梯度、clip、Adam状态、非空CPU反序列化恢复与下一更新一致、导出回读、公共角平移/同步翻转D与phi、单次11点初始化及未来输入拒绝。通过与否仍待实际CUDA QA。
- cost恰三源各四次更新计算：1、2参照、恢复1后重算2、3；没有错误调用只接受32倍数的旧restore。专用cost快照明确不作科学起点，保存实际更新数为3及cost_only身份。
- 费用估计逐源使用最慢完整更新时间、17次保存、载入、三个里程完整目标、25%余量，分别判断640/40；没有用三源均值掩盖慢值。完整评价计时使用独立非科学16×61与16×111观测，包含teacher双步长工作量、模型双步长评价、逐事件结果和写盘；不提前生成正式stream1。
- 正式train从共同未拟合source开始，固定512更新及stage0/每32次保存，共17点。三个512端点在任何新train选择前统一核身份和哈希；按完整train MSE、已锁定AST节点数/系数数/规范字符串逐源排序；formal val路径要求全部三源冻结清单及端点字节不变。复杂度7/5在tests中明确核定。
- helper此前四项修改已落实：显式native本地AST依赖闭包及consumer独立核验；QA/cost共享不可变科学输入的实际before/after/current身份；实际complete PASS及其哈希；调用native后缺少真实GONE证据保留锁；phase_gate失败返回非零。540/570/600、独占输出、单次QA后单次cost及2200互斥分项保持。

## 成本与后续边界

runner的train和eval估价各加入固定20秒启动/退出预留。这是预登记的保守预留，不能写成已测启动时长。root须用已有同环境启动证据确认它覆盖import/bootstrap/exit和末端JSON写入，或在唯一cost的原生回执中核对；若不覆盖，应按原费用准入失败处理，不缩任务/重测/改门槛。

P1 CLI代码存在不等于P1已授权。后续仍须独立的P1登记、固定命令/源与端点绑定、预算分项和监督回执；本轮P0 helper只允许qa/cost。原合成val已知，结果只能支持登记两种结构的条件性预测辨别；不认定未知恢复、唯一隐藏坐标、真实电网新模型或P4资格。

审查时root提供身份：runner `7c57419183d9f0f3dd18b3884df72bbd4deca108b810cd6795e221b4b07875e9`，tests `155364a3a8dcb7054fc52c08a590befa3234d9118a28f69703f5e286044eddb9`；helper make `45634634a8718ddc82a412154279a55d70c81dda27a1213391346c260195fa34`、launch `9745e06095f1333f0ddc650fc17844a4daf1cd7ca9af3654fa49544a34330a69`。这些为root转交身份，最终精确身份以root已核冻结表为准。修正后的新身份须重新归档，不能沿用修正前源码名义启动。

计费：本次前三个只读原生命令为2.0014100、1.8779492、2.0618062秒，合计5.9411654秒；本次写审查文件的回执另计，由root合并实际值。第一条输出显示截断，第二条已补读完整cost/train/freeze/evaluate及tests。审查工具累计上限15秒，属于P0支持700；与此前1.8280482秒helper审查分别记账。

