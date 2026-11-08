# train fit297 主线接入准备

Run 3 可复用现有 train-only 入口和 L1 统计定义。当前尚未读取事件数据或释放训练；以下为源码与静态元数据核查，提供者为 root、Modeler 与独立 Reviewer。

- **数据**：`experiments/closure_probe_20261030.py` 的 `load_frozen_events` 经 `checked_preparation(..., fit_only=True)` 核对 297 个事件、267 稳定及 30 失稳。不得改用会加载 no_update 的分支。`configs/dataset.json` 所指旧 canonical 库不是 fit297 执行入口。
- **模型**：GFTD 为 5N；原 `r_memory_study_20261025.evaluate_model` 假定 Torch 模型、encode、p_scale 和 40 维失败占位，不能直接接入。适配应输出 theta−delta_star、omega、p，并保留实际 5N 状态和原分母。同期对照可使用 GFTD-static；历史 R_N 属于 ForceGraphROM，不能未经映射核查套用只接收 RMemoryROM 的 from_parent。
- **L1**：复用原 `_point_error`、risk、分类及双步资格。risk 是逐事件归一化点平方误差的 0.8 倍均值加 0.2 倍最差 ceil(10%) 均值；共同转速 RMSE 单独用 rad/s，p 单独用归一化 RMSE。8 s 与真实可用范围分类分别报告；历史 21/30 不作为本轮同期成绩。
- **L2**：只消去常数全局相角，保留共同 omega、p、r。已有采样稳定域代码不能代替同一 5N ODE 的证明，完整局部条件见调制准备稿。
- **L3 模态**：可复用 `src/kiki/evaluation/modal.py` 的 `state_space_modes`、MAC 和 `match_dmd_modes`。现有 `finite_difference_augmented_modes` 同时删除共同相角与共同转速，不能直接套用；需对实际 5N Jacobian 作正确角度商空间处理。
- **L3 物理参考**：旧 `data/kiki_ieee39_v1/manifest.json` 标记 `ieee39/ieee39_full.xlsx`、SHA256 `9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5`、ANDES 2.0 和 GENROU_1..10 的相角顺序。指定旧 `.venv` 及当前 `.venv-gpu` 未找到 andes/cases；可访问的已引用资产中也未找到实际算例文件。物理 Kron 矩阵、GENROU→bus 映射、内部电势定义及负荷消元假设仍为 unknown。未构造矩阵；有效图不能冒充物理导纳。

Kron 对照需要找到并核验实际算例或已保存矩阵。节点电压 Kron 与内部电势 Kron 是不同参照，后者还需冻结暂态电抗、内部电势与负荷等效假设。该缺口不阻止前两步的人工研究及接口实现。
