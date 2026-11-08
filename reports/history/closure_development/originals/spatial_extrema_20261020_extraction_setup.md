# O/S空间极值研究：结果提取准备

2026-10-20。新增`experiments/extract_spatial_extrema_20261020.py`，从六个512步角色的同次评价JSON提取共同385、原366及新19的指标。cohort身份绑定既有SHA256，保留342条稳定事件和43条失稳样本，不按名称前缀划分。

提取前要求root传入冻结manifest的实际SHA256。程序先核对六角色、登记路径、512步、来源hash格式、schedule身份和旧6101_O固定三个hash，再核对六个完成标记及evidence hash；全部事件及provenance与manifest匹配后才计算汇总。旧6101_O的模型引用保留在temporal目录，其评价队列位于本轮spatial目录。程序不读取模型、training.pt或checkpoint成绩；来源训练语义由生产端冻结前的`check_endpoint`核验承担。

原点误差、稳定5 s风险、共同频率rad/s及机械功率100 MVA基准pu/MW、实际15 s三项失稳误差、15 s与8 s独立分类均保留。物理RMSE合并实际SSE和标量点数。空间解释指标要求实际15 s及pair15完整数值资格，并核对其已保存的资格和finite；相关事件失败时总体为null，保留分母和事件ID。

COI归一化RMSE独立列出：mean为事件RMSE算术均值，pooled为`sqrt(sum(normalized_sse)/sum(n))`，worst10为最差ceil(10%)事件RMSE的均值。数据来自原`records`并join原`numerical`及8 s资格，不把逐点归一化平方误差称为RMSE。与原汇总一致，不可用时域单列数量；合格判断不使用finite子集替代总体。S−O输出配对事实，不判断科学采用或显著性。

事件级触发、资格、标签、母线、原分量与空间指标保存在私有目录`spatial_extrema01/extracted_event_rows_step000512.json`；39维局部数学结果只摘录已有local.json，不重新计算Jacobian或给出区域结论。

五项stdlib unittest合成测试通过，测试自身0.160 s，原生命令3.0048702 s。覆盖旧控制路径、冻结角色、同前缀集合、单位与SSE池化、风险、资格和分母、finite不足及COI三种聚合。测试不依赖私有资产；本次未读取任何新训练/评价成绩，没有运行实际提取。等待六评价完成并归档后由root执行：

```powershell
& 'C:/Users/LLLzijian/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' experiments/extract_spatial_extrema_20261020.py --manifest-sha256 FROZEN_SHA256 --output NEW_JSON
```
