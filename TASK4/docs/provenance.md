# 来源与本次新增内容

已认可科学设计：`docs/references/TASK4_design_20260929.md`，与此前用户收到的TASK4.md原文一致。
数学附录：此前TASK4_design.zip中的architecture_math.md。
KIKI数据schema与坐标：用户提供loop101.zip中的train_local_memory_0098.py、private_dataset.py、private_pm.py、private_rom.py；本工程独立重写只读适配器，不复制数据和权重、不直接import KIKI。
治理新增：根AGENTS、工程loop、审批边界、接受与回退规则，落实本轮用户要求，优先于科学设计中旧的执行建议。
代码新增：本工程的PyTorch初始模型、训练/压缩组件与状态工具。它们是可运行起点，不是来源报告已验证的新电网成果。详细实现范围见implementation_status。

当前目标文件夹：`parent/TASK4`；源文件夹：`parent/kiki`。无系统路径硬编码。

T4_0001附加原13模态评价的只读来源：KIKI `experiments/evaluate_local_memory_0098.py`中的固定目标、缓存核验、Jacobian QA；`experiments/compare_physical_modes.py`中的原物理输出尺度和有未匹配选项的Hungarian配对。计划独立移植至TASK4并以合成用例验证，运行时记录源代码/固定缓存哈希，不从KIKI导入Python模块。固定平衡点模态缓存为`private_data/task2_control_modes_0078/{evidence.json,modal_arrays.npz}`；不重新计算FOM，不读取事件载荷。实现与实际执行完成情况分别见当前状态及本轮报告。
