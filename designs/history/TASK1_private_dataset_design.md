# 任务一：私有 IEEE-39 后故障暂态数据集设计方案

## 1. 任务目标

在当前项目目录内建立一套专门服务于后故障低阶动力学发现、记忆建模与稳定性分析的私有数据集。

数据集的基本对象不是“故障参数”，而是由不同扰动产生的、物理可达的 post-fault 初始状态及其随后在同一系统中的自由演化：

\[
 x_0^{(i)} \rightarrow x^{(i)}(t),\qquad \dot x=F_{post}(x)
\]

所有 accepted events 必须共享同一个 post-fault 系统。无论采用何种故障位置、故障持续时间或其他短时扰动，故障清除后都必须恢复为同一套网络拓扑、元件参数和控制配置。

数据集首先以 Pilot 形式建立，验证数据生成逻辑、关键变量保存、\(P_m\) 动态和模型可用性；Pilot 通过后再按相同规则扩展为正式数据集。

---

## 2. 数据集定位

该数据集用于：

1. 建立统一 post-fault 系统下的低阶暂态动力学模型；
2. 研究短时 \(\delta,\omega\) 历史对隐藏高阶状态的重构能力；
3. 识别显式 \(P_m\) 动力学；
4. 为稀疏方程、稀疏能量函数和稳定性分析提供训练与验证数据；
5. 保存必要的 full-order 状态，用于解释 latent/memory state 的物理含义；
6. 验证发现的 reduced model 能否跨不同扰动来源统一描述同一 post-fault 系统。

故障线路、故障位置、故障持续时间等只作为**初始状态生成条件和数据分组信息**，不作为最终 reduced model 的输入。

---

## 3. 目录与私有性要求

数据集保存在项目自身目录下，并默认不进入 Git。

建议使用简洁结构：

```text
project_root/
├─ private_data/
│  └─ ieee39_postfault/
│     ├─ dataset.h5
│     ├─ dataset_index.csv
│     ├─ splits.json
│     └─ qa_report.md
├─ configs/
│  └─ private_dataset.json
├─ experiments/
│  ├─ generate_private_dataset.py
│  └─ audit_private_dataset.py
└─ reports/
   └─ private_dataset_build_report.md
```

`.gitignore` 必须包含：

```text
private_data/
```

项目仓库只提交：

- 生成脚本；
- QA 脚本；
- 配置；
- 不含原始轨迹的统计报告。

Pilot 数据链验证通过后建立一个 Git checkpoint，固定生成脚本、配置、QA 脚本和数据接口；正式扩展沿用该 checkpoint 的数据定义。


---

## 4. 基准 post-fault 系统

### 4.1 唯一基准系统

首先冻结一个唯一的 IEEE-39 动态系统配置 \(\mathcal S_{base}\)。至少固定：

- 网络拓扑；
- 线路和变压器参数；
- 负荷模型；
- 发电机动态模型；
- AVR、PSS、governor、turbine 模型及参数；
- MVA base 和频率 base；
- 基准潮流与调度状态；
- TDS 积分器和主要数值设置；
- 故障清除后的网络开关状态。

配置文件中记录 ANDES 版本、case 文件、关键模型配置和生成脚本版本即可。

### 4.2 所有样本恢复相同 post-fault 系统

对任意 accepted event：

\[
\mathcal S_{post}^{(i)}=\mathcal S_{base}
\]

必须满足：

- 故障元件被移除；
- 临时参数变化被恢复；
- 网络拓扑恢复基准状态；
- 不保留永久切线或永久切机；
- 控制器结构与参数一致；
- post-fault equilibrium 对所有事件属于同一系统。

生成脚本必须在保存事件前执行一致性检查。若扰动无法恢复到基准 post-fault 系统，则该事件不进入数据集。

---

## 5. 扰动设计原则

### 5.1 扰动只用于产生不同 clearing states

数据生成的目标是覆盖尽可能丰富的物理可达：

\[
x_{clear}=[\delta,\omega,P_m,P_e,x_{hidden},y_{alg},\ldots]
\]

故障本身不是最终研究对象。

### 5.2 优先增加状态覆盖，不做密集 CCT 扫描

不围绕 CCT 做毫秒级 dense sweep。优先通过以下维度产生不同 clearing states：

1. **故障线路**：覆盖多个电气区域和不同潮流强度线路；
2. **故障位置**：例如线路 20%、50%、80% 等少量离散位置；
3. **故障持续时间**：使用少量离散时长，例如 50、100、150、200 ms；
4. **故障类型**：Pilot 优先使用稳定、易控制且可恢复系统的故障类型；
5. **可选短时扰动**：仅在能保证清除后恢复基准系统时使用。

第一阶段不为了样本量扩展过多扰动种类，优先保证每一种扰动都能正确保存完整状态和 post-fault 演化。

### 5.3 运行点

Pilot 阶段固定同一个基准运行点，避免把“不同 operating point”与“不同 hidden state”混在一起。

若 Pilot 和算法主线验证通过，再把 operating-point variation 作为正式数据集扩展项，并明确分组，不改变第一阶段的研究基准。

---

## 6. 仿真时间设计

建议统一：

- disturbance 前先确认系统处于稳定基准状态；
- disturbance start 使用统一参考时刻；
- clearing 后立即恢复 \(\mathcal S_{base}\)；
- post-fault 最大记录窗口建议 10–15 s；
- 默认采样间隔 10 ms；
- physical unstable event 可在明确的失稳判据触发后结束。

15 s 只是最大观测窗口，不要求失稳轨迹补齐到统一长度。

---

## 7. 失败与发散样本处理

这是数据生成的硬约束。

### 7.1 数值失败直接丢弃

出现以下任一情况时，该事件直接 `continue`：

- Newton/TDS 不收敛；
- solver error；
- NaN/Inf；
- integration step collapse；
- 状态突然冻结；
- 尾部重复最后一个有效值；
- 数据长度或时间轴异常；
- 无法确认发散属于真实物理失稳；
- 任何其他数值失败或导出异常。

这类事件不保存为 unstable，不 padding，不截取前缀进入正式数据。

### 7.2 只保留两类 accepted events

#### Stable

solver 正常工作并积分至规定 horizon，系统满足稳定判据。

#### Physical unstable

solver 仍正常工作时，系统满足预先定义的 loss-of-synchronism / physical instability 判据。

可在判据触发后结束事件：

\[
T_i=T_{LOS}
\]

### 7.3 终止原因

只允许：

```text
stable_horizon_reached
physical_loss_of_synchronism
```

数值失败事件不进入 dataset_index。

---

## 8. 物理失稳判据

必须把“物理失稳”和“solver 失败”分开。

Pilot 阶段使用简单、统一、可复现的判据，建议至少结合：

1. generator rotor-angle separation；
2. sustained speed divergence；
3. 判据持续若干连续采样点，避免瞬时噪声触发。

例如可采用：

\[
\max_i\delta_i-\min_i\delta_i>\Delta\delta_{LOS}
\]

并要求持续 \(N_{hold}\) 个采样点。

具体阈值写入 `configs/private_dataset.json`，在 Pilot 生成前确定，生成过程中不根据结果修改。

---

## 9. 必须保存的变量

### 9.1 核心变量

每台发电机至少保存：

- \(\delta_i(t)\)；
- \(\omega_i(t)\)；
- \(P_{m,i}(t)\)；
- \(P_{e,i}(t)\)。

这些变量是算法主线的必需数据。

### 9.2 Full-order 动态状态

尽可能保存 ANDES 中与同步机和控制器相关的动态状态，包括：

- transient / subtransient EMF；
- AVR states；
- governor states；
- turbine states；
- PSS states；
- 其他实际参与 TDS 的 dynamic states。

同时保存一张简单的 state mapping：

```text
column_index
state_name
model_name
device_id
```


### 9.3 网络代数变量

至少保存：

- bus voltage magnitude；
- bus voltage angle；
- generator terminal voltage；
- active/reactive power；
- 后续能量和 DAE 一致性分析需要的关键网络量。

---

## 10. 坐标与基准

HDF5 中优先保存 ANDES 原始状态；算法读取阶段再生成统一 canonical representation。

至少支持：

- 原始 \(\delta,\omega,P_m,P_e\)；
- COI projected \(\delta\)；
- COI projected \(\omega\)；
- COI projected \(P_m\)；
- COI projected \(P_e\)；
- equilibrium-relative coordinates。

所有变换统一写在一个数据 adapter 中，避免数据生成脚本和训练脚本各自维护一套变换逻辑。

---

## 11. 数据结构

建议只使用一个 HDF5 主文件和一个索引 CSV。

HDF5：

```text
/events/<event_id>/
    t
    delta
    omega
    Pm
    Pe
    full_state
    algebraic_state
    equilibrium
```

`dataset_index.csv` 每个事件一行，保留最小必要 metadata：

```text
event_id
fault_line
fault_location
fault_type
fault_duration_s
clearing_time_s
label
termination_reason
event_time_s
trajectory_end_s
valid_length
```

必要时增加 `disturbance_family` 或 `operating_point_id`，但不提前设计通用 metadata framework。

---

## 12. 数据集拆分

禁止 trajectory-level 随机拆分。

Pilot 阶段使用简单的 group split：

- train；
- validation；
- holdout test。

优先按 fault line / disturbance family 分组，使测试扰动来源不出现在训练集中。

`test` 一旦确定后不参与模型结构选择。

Pilot 数据较小时，可先采用一个固定 development split；正式扩展后再增加 grouped cross-validation。

---

## 13. 数据质量门

`audit_private_dataset.py` 至少检查：

1. post-fault system 与 baseline 一致；
2. 无 NaN/Inf；
3. 时间严格单调；
4. sampling interval 在容差内；
5. 无 frozen tail；
6. 无重复 padding；
7. \(\delta,\omega,P_m,P_e\) shape 一致；
8. full-state 列数在所有事件中一致；
9. stable/unstable 标签与统一判据一致；
10. unstable 事件在触发 LOS 前 solver 正常；
11. COI 变换残差满足数值容差；
12. train/val/test group 不泄漏。

QA 脚本输出一份简洁 `qa_report.md`，列出通过数量、丢弃数量、失败原因统计和关键数值范围。

---

## 14. 去重与有效信息量

不以事件总数为唯一目标。

Pilot 阶段只做简单 coverage 检查：

- clearing-state 在 \(\delta,\omega,P_m\) 空间的 PCA；
- 最近邻距离；
- stable/unstable 数量；
- 不同 fault line / fault location 的覆盖；
- clearing-state 分布是否高度聚集。

若一批事件几乎重复，可在正式扩展时减少对应扰动组合。

无需在 Pilot 阶段建立复杂的自动去重框架。

---

## 15. 数据生成流程

```text
加载唯一 baseline IEEE39 system
        ↓
读取 disturbance list
        ↓
for disturbance in list:
        ↓
恢复 baseline
        ↓
确认 pre-disturbance steady state
        ↓
施加短时 disturbance
        ↓
clearing 后恢复 baseline topology/parameters
        ↓
检查 post-fault system consistency
        ↓
运行 post-fault TDS
        ↓
numerical / solver failure ?
    yes → continue
    no
        ↓
physical LOS ?
    yes → 正常结束并保存
    no  → 跑到 horizon 并保存
        ↓
事件级 QA
        ↓
通过 → 写入 dataset.h5 + dataset_index.csv
失败 → continue
```

生成脚本必须支持固定随机种子和配置化 disturbance list，使数据可以重跑。

---

## 16. 数据规模与阶段安排

### Stage 1：Pilot dataset

目标是尽快验证整条数据链和算法可用性。

建议：

- 目标 accepted events：约 300-400。

Pilot 必须同时包含：

- stable events；
- physical unstable events；
- 足够不同的 clearing states；
- 可观察到真实 \(P_m(t)\) 变化的长时稳定轨迹。

Pilot 的目的不是形成最终统计规模，而是确认：

1. post-fault system 恢复逻辑正确；
2. full-order state、\(P_m\)、\(P_e\) 保存正确；
3. numerical failure 能可靠排除；
4. history-memory 和 \(P_m\)-ODE 实验可以在新数据上运行；
5. 数据字段足以支撑 M3/M4 与能量结构研究。

### Stage 2：正式扩展

只有 Pilot QA 和任务二前两个 milestone 通过后扩展。

建议正式规模：

\[
N\approx 500\text{–}1200\ \text{accepted events}
\]

扩展优先增加：

- 新 fault lines；
- 新 fault locations；
- 少量新的 duration；
- 必要时再加入新的 operating-point family。

扩展目标仍然是 clearing-state coverage，而不是机械追求样本数量。

---

## 17. 必须输出的产物

Pilot 阶段完成后至少有：

```text
private_data/ieee39_postfault/
    dataset.h5
    dataset_index.csv
    splits.json
    qa_report.md

configs/private_dataset.json
experiments/generate_private_dataset.py
experiments/audit_private_dataset.py
reports/private_dataset_build_report.md
```

正式扩展继续沿用同一格式，不另建第二套数据框架。

---

## 18. 任务完成标准

### Pilot 完成标准

- 所有 accepted events 的 post-fault system 一致；
- numerical failure 全部被直接丢弃；
- 无 frozen tail、last-value padding 或伪失稳；
- stable 与 physical unstable 均存在；
- \(\delta,\omega,P_m,P_e\) 完整；
- full-order dynamic states 可读取且 mapping 正确；
- 数据可由一个生成脚本按配置重复生成；
- QA 脚本全部通过；
- train/val/test 不存在 group leakage；
- 数据能直接进入任务二的 M2/M3-Pm/M4 实验。

### 正式数据集完成标准

在 Pilot 标准基础上：

- clearing-state coverage 相比 Pilot 明显扩大；
- 主要 disturbance groups 均有代表；
- 样本规模足以支撑 grouped validation；
- 最终 holdout test 保持冻结；
- 生成和 QA 流程不因扩展而改变核心数据定义。

---

## 19. 研究口径

该数据集定义为：

> **由不同短时扰动生成、共享同一 post-fault 系统的物理可达状态与自主暂态轨迹数据集。**

核心研究对象是：

\[
\boxed{x_{clear}\rightarrow \text{post-fault autonomous dynamics}}
\]

扰动负责产生不同初始状态，真正需要被发现和分析的是统一的 post-fault 动力学结构。
