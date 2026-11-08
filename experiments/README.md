# 实验代码入口与历史恢复

新科学比较按 `experiments/<run_id>/run.py` 保存入口，需要独立后处理时再建 `analyze.py`。公共模型、初始化、积分和评价实现在 `src/kiki/`；入口负责组织比较、来源冻结与执行回执。

## 共享实现

- 现有模型与 GFTD 推断实现从 `kiki.models` 使用；模型特定的部署和数值合同沿冻结协议。
- 新的通用 NumPy 自治 RK4 可直接导入 `from kiki.numerics import rk4_step`。它支持向前／向后一步，调用者负责状态坐标、投影、步长和数值资格；库不加载数据或模型。
- 旧执行继续绑定原源码和完整原版本。新公共模块的等价性检查没有重算任何旧实验，也不把旧释放清单回填成新模块。

## 原位保留与历史归档

[当前代码清单](../results/repository_organization/20261109_history07/code_index.csv)覆盖整理前238个顶层Python脚本，列出入口／共享实现线索、具体调用者及路径处理。29项已归档，209项有依赖或明确保留理由；旧清单的22项已逐项复核。没有调用线索不等于可以删除。

[早期 E 系列代码](history/early_e_series/README.md)保存已闭合入口。其 `ROOT` 与导入可能依赖旧目录深度，应按说明恢复完整原提交后阅读或复现，不直接在更深的归档位置运行。

归档和公开数值模块不释放训练、原val/test、新FOM或历史作业重跑。实际执行仍由[当前状态](../docs/current_state.md)和相应协议决定。

[TASK2历史入口](history/private_rom/README.md)、[已知支撑对照](history/closure_discovery/README.md)与[力记忆检查](history/force_memory/README.md)分别按原问题查找，保留完整原版本恢复方式。
