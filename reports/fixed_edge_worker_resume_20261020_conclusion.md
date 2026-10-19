# 未来worker的step0恢复接线已修正

2026-10-20。真实worker从安全step0恢复时，现在复用已经发布的检查点；新运行仍首次发布一次。唯一合成验收通过，原生完整包络57.5088199秒、exit0、无超时，64冻结文件未变。独立审查同意把恢复能力修正作为工程进展纳入有限PR；默认模型和已完成六模型的科学结论保持。

## 修改与实际验收

实际gpt-6-astra规划/root/astra6_after_fixed_pair，完整原始返回与root采纳见[协议](fixed_edge_worker_resume_20261020_protocol.md)。原P2结项73b5b0677b96cb9e7915e836222cd6d2ee247494后，独立登记repair1800秒，一次QA1200加保守支持600，不续用原两次QA或科学恢复机会。

修正只有四处语义修改：train_steps新增save_initial=True，首次保存0受此条件控制；worker恢复重放和恢复后的普通入口显式关闭首次发布。其余保存、采样、优化器、数值规则与身份检查不变。没有允许覆盖、按文件存在跳过或捕获覆盖错误继续。

| 固定案例 | 实际结果 |
|---|---|
| 冻结旧源码的实际worker负对照 | 真实触发cannot overwrite the active immutable checkpoint，step0及latest未变 |
| 新worker首次运行 | 首次step0恰好发布一次，正常执行两次真实短合成更新 |
| 安全step0恢复target0 | 零重放后普通入口成功，与连续运行的参数/Adam/RNG/覆盖/trace一致 |
| 安全step0恢复target1 | 实际重放一次，随后继续；上述连续性与不可变step0通过 |
| 安全step50恢复target51 | 重放一次后更新52，与连续路径一致；marker顺序、<=50步重放及latest保护通过 |
| 错身份、model/training文件损坏、不安全保存与第50步失败 | 全部拒绝，不推进latest或发布失败状态 |
| 原生正常、非零及超时进程树 | 回执和实际worker一致，非零保留7，超时自有子树清理通过 |

验收实际调用worker，保留原save_safe、restore_cached、严格身份adapter、采样器和保存回调。合成来源/预检替换只在测试进程，未读取S-O真实来源、科学检查点、train/val/test或FOM。CUDA float64；模型/Adam原容差atol=rtol=2e-9，整数/身份/随机流精确。长案例第2–50步不更新模型，只检查采样、trace和保存控制流；共14次真实短合成更新，不把它称为完整1024恢复试验。QA工件不作为科学候选。

独立审查/root/fixed_edge_final_review未参与修正或测试实现，静态审查后又只读固定JSON及原生回执，确认7项全部通过，原阻断在未来版本关闭；同意范围受限工程PR。root采纳，最终仍须核对精确PR文件与预期head及远端合并。修正不追改原P2的旧QA分类；旧runner和全部不利证据保留原归档。

## 费用与复现

唯一完整原生包络57.5088199秒，含子测试、启动、退出与清理；内部Python56.3866102秒和native子测试费用均已包含，不重复加。监督53312、launcher87848、实际worker26332均已退出。来源94ccfd0e1a2fa408032bed715123958c44ab33f1。支持保守600，结项657.5088199；未用1142.4911801明示取消，不自动再测。

父累计实际加保守支持51472.1208810／259200，余207727.8791190；repair剩13407.9059650。其他类别余额及旧900支持hold保持。原P2六模型不重跑、没有新模型采用，原test关闭、TASK4暂停。

```powershell
.venv-gpu/Scripts/python.exe tests/test_fixed_edge_worker_resume_20261020.py --output private_data/closure_mechanism_20261019/repair/worker_resume01/qa01 --max-seconds 1200
```

该目录已完成，禁止重复使用；复现须在另有授权的新目录执行。完整Stopwatch/自有进程树监督源及哈希、原生回执、实际case摘要和完整本地qa.json的路径/哈希见[紧凑证据](../results/worker_resume_20261020/summary.json)。原生监督计时含全部子进程包络。旧负对照可用本地legacy副本，或在完整Git历史读取47ba6f341886aeb4cc235d3c25a9b16bc8f4a24b的旧runner blob，严格校验SHA4412777530c9a40f80641faf431991e2c62f38194c15023c8d6f1417ce4c14f9；不需私有科学数据。

runner SHA6f3194387f13a103a04bd73d27dbe17cde7e786d8617655ba2fff8bf811d72c3，QA SHAefa100e889f7711c39a02f42ca96fff411b5217e2f4d82706ac5edd522b3a31b。旧科学复现使用47ba6f...，不能把未来版本哈希替换到旧证据中。下一科学阶段仍需实际Astra6分析与有限登记，本修正不开放P3采集、生成搜索或P4区域计算。

PR11已合并并核实：预期head b4a04b1bf326108114ca9e975b0698a1fe682765，远端main51370332dd68054028fdeeb86d967c8cc5b72ffb。15个有限文件，保留完整旧源祖先；无远端CI任务，实际本地唯一验收及独立审查通过。默认模型不变。
