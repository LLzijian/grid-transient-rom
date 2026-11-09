# GFTD prepare02 路径修复审查

**接受修复及 prepare02 生成的选择清单。** 首次 prepare 在 `static_000` 的存档状态路径比较处中断。旧路径用 Windows 分隔符，程序用正斜杠写预期路径。修复把旧路径转换为统一文本后再比较。实际读取仍使用预期的仓库内相对路径，并逐字节核对冻结清单中的 SHA-256。源码差异限于这处比较和第二次 prepare 的释放、启动凭据及回执。诊断算法沿用原审查。 [修复源码](../../experiments/gftd_initialization_precision_20261109/run.py#L263) · [原实现审查](diagnostic_implementation_review.md)

- **首次中断记录。** 原释放、启动凭据、执行和宿主回执均与保全副本逐字节一致。原执行回执 SHA-256 为 `0e5b38467004ca51cfefa75eb71849f101e27ea6c21eca50f37900240e31a6fd`，状态为 `INTERRUPTED_RUNTIME`，已完成项为空，未生成选择清单。源码的 prepare 路径不进入 RHS 诊断。 [首次执行回执](../../results/gftd_initialization_precision_20261109/prepare_execution.json)
- **第二次启动与实测结果。** prepare02 绑定原中断回执的哈希，要求原阶段中断、已完成项为空、新 action 不同且选择清单尚未生成。新启动凭据独占写入，原文件保留。已完成的 prepare02 回执与释放、启动凭据一致，选择清单为 755964 字节，实测 SHA-256 与回执一致，为 `24b6c3f9650374bc4e6d9b6b62c512d82c489cad8b30282e78819048bd8e3679`。该清单可绑定到独立的诊断释放。 [prepare02 释放](../../results/gftd_initialization_precision_20261109/prepare02_release.json) · [prepare02 执行回执](../../results/gftd_initialization_precision_20261109/prepare02_execution.json)

核对的修复源码 SHA-256 为 `25fa1adc10aabaf7411bfe2a7069fcd7f292d0e78c9e274f473deef3d6286873`，测试源码为 `9c445cd08c9d31180ccae1800dd7d983425628dbac6dd21fa620052e91a83893`。沿用 8 项人工测试通过回执。本次审查核对源码差异、回执和选择清单哈希，未重复测试或读取私有载荷。 [人工检查回执](../../results/gftd_initialization_precision_20261109/prepare02_qualification.json)
