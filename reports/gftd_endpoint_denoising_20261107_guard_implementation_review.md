# 同点守卫纠正工具：独立实现审查

结论：冻结实现符合严格同点纠正设计，未发现阻断项。状态为 `eligible_for_one_zero_integration_postprocessing`，供root释放一次纯后处理；不释放确认实验，不预先认定纠正后的完整采用条件通过。

冻结输入：

- 工具 `tools/gftd_endpoint_guard_correction_20261107.py`：SHA256 `5E081BCEB5675224404FE951B1FC8006BBC6AAB0274499EE63B56F5763179F95`。
- 测试 `tests/test_gftd_endpoint_guard_correction_20261107.py`：SHA256 `CEDDC6BC76FF3D2C36D5FBDB06A2FE5793DE6EC1C7F151EB45D26E1620C43B31`。
- 修复设计：`A66A3F15919233A5B6F89867063EDDD5A0E21B4858BA406D893699779E34051F`；设计审查：`2BF5F2E0AC2F1B08FCCF4F765DC029EFD83584CC4FDAF24577984883E3350F31`。

## 实现核对

1. **证书独立于预测评分。** 第62–109行仅接收拟合点、父点、合法观测与源谱系证据；按binary64位、shape及有限性核对完整state、坐标、初值、父可行点、步长、尺度与同一步长损失。无allclose或ULP准入阈值。父status收紧为converged，以绑定实际成功返回时的重建raw算术路径。未来指标未进入身份证书。
2. **独立保留其他原门。** 第112–138行另核父reference及endpoint资格标志、往返/历史/未来主细数值门、p逐值为0、保存的预测判定与RMS/peak重新判定一致。预测失败可返回 `passed=True, classification=prediction_fail`，不会因不利分数拒绝身份；资格或主细判定冲突保持未决。没有fine的真实记录符合本次有界处理范围。
3. **扫描与变更范围封闭。** 第157–221行扫描全部256个main；预期四个ID用于发布影响范围校验，未进入身份谓词。额外同点false guard即使另有数值失败、不能改变分类，也会暂停发布。只有四条anchor分类及六项依赖汇总可变：恢复这些字段后必须与原summary完全相同；高噪均值还另行核对不变。配对分类计数和逐输入结果在附录另列，不改原comparisons、worst或任何预测指标。
4. **采用条件使用冻结定义。** 第141–153行调用原冻结adoption_conditions纯函数，重算主/细及有无细步适配的四组条件和一致性。已检查其导入链，加载定义不会调用RHS、积分或拟合；本次调用只聚合保存数值。原worth=false独立保留，不能以手工改总开关替代全部条件。
5. **证据与独立产物绑定。** 第232–271行核两代释放与完成回执、源文件、fits、summary、共享观测及父fits关联；共同固定参数工厂调用由AST比对。主入口另核修复设计和root释放清单，独占创建started标记，输出证书、有效分类附录及完成回执，前后复核原证据及纠正源未变。新附录明确confirmation_released=false。root释放清单应绑定本工具、测试、设计/审查及两代源、release、receipt、fits、summary、obs、truth哈希。

18项纯模拟JSON资格测试由root/Modeler报告通过；Reviewer读取最终测试，核到逐位差异拒绝、预测失败不变成功、父budget_limited不冒充重建路径、全256范围、额外身份影响暂停与原证据不变等关键分支。Reviewer未重复运行测试或真实audit。

代码决定：可释放一次零积分、零拟合后处理。模型决定：不晋升。科学决定：等待附录落盘后独立复算四条证书、完整132条判读及全部采用条件；此处不宣布纠正完成，不新增实验成功证据。具体1 ULP舍入成因未逐元素重演，修复依据仍为同点数学身份。
