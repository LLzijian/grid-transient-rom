# 研究材料导航

从[报告首页](../reports/README.md)按中文问题进入，主题认识维护在 Campaign，科学身份与决定仍来自 `harness/runs/` 和所引用的原件。

## 查一个 Run 或全部登记

```powershell
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py show-run --root . --all
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py show-run --root . --run gftd_fit297_20261107
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py check --root . --run gftd_fit297_20261107
```

默认只输出终端。`--format json` 提供字段级材料、原登记状态、未知项和冲突；`--output <新文件>` 显式生成导航，拒绝覆盖已有文件。更新已生成页时，先输出到同目录的新文件，对照原页后替换；不要用导航生成器覆盖冻结报告。相同输入生成稳定内容，不添加调用时间。

`check --root .` 继续执行原有全仓库公开元数据检查；`check --run` 定向检查该 Run 的明确引用与导航事件，二者覆盖范围不同。

## 信息分别从哪里来

| 信息 | 唯一维护来源 |
|---|---|
| 科学身份、范围、执行记录状态与三个决定 | 现有 `runs/<run_id>.json` 及其证据；旧字段没有记录时显示 unknown |
| 中文阅读标题、主题、父关系、明确材料用途、更正和技术续行关系 | [navigation.json](navigation.json)，标注追溯整理，不复制指标和科学状态 |
| 当前跨 Run 认识、竞争解释与未决问题 | 相应 Campaign；直接引用结论、审查和证据 |
| 无正式登记的历史组 | [legacy_index.json](legacy_index.json)与现有主题材料表，不补造历史 Run／Job |
| 所有已登记迁移的当前位置 | [current.json](../results/repository_organization/current.json)固定当前输入版本，共用解析器验证旧文档和累计归档目标的原始字节 |
| 保护、代码、存储及职责复核范围 | [当前生成入口](../results/repository_organization/CURRENT.md)与`show-organization`；历史批次继续保存原时点 |

原执行、技术续跑、root 决定和后续归档分别显示原回执与时间字段，不按文件修改时间选一份“最新真相”。技术续行不计为独立科学重复。一次决定中的待归档动作与稍后的归档完成可以同时成立。

平铺引用没有明确材料用途时保留字段来源，不根据文件名赋予结论或确认身份。来源缺失与来源内容错误是两种检查；本工具只核查前者及记录内部的显式状态冲突。

## 路径检查的含义

- `present`：公开路径存在；不表示证据内容或科学结论已通过核验。
- `migrated_sha256_verified`：已按当前输入指定的文档／报告／代码／JSON映射核对目标大小与原始字节SHA256。
- `local_not_checked`：本地数据、模型、权重或产物的引用保持不透明，没有打开或核查其存在性。
- `missing`／`unsafe`：公开引用缺失或路径不安全，命令返回非零并列明来源字段。

只打开登记、导航和明确列出的少量执行／决定／归档元数据；不递归跟随回执读取科学数组或权重。保存状态不核验实时进程，任何导航输出都不释放执行权限。

闭合更新的步骤见[工程循环](../docs/engineering_loop.md)和[闭合模板](../templates/run_closure.md)。新材料应从创建时按职责保存；历史冻结记录不批量回填。

## 当前整理状态与维护

```powershell
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py show-organization --root .
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py show-organization --root . --path reports/0102_prediction_optimization_capability.md
./.venv-gpu/Scripts/python.exe -X utf8 -B tools/research_harness.py show-organization --root . --path results/e08 --format json
```

默认只读输出。原路径和当前位置均可查询；结果区分所属范围、用途核实程度、保护消费者、具体解除条件及连续阅读／原件入口。`in_git`表示该文件或目录下至少一个文件被跟踪，`local_presence`只检查路径本身，`remote_availability=not_verified`不承诺远端完整取得；原表的`public_results`名称不是公开可取得的证明。

`current.json`只维护输入路径及`sha256_lf`版本，不复制科学状态。导航输入明确按LF规范化核验，迁移原件仍按原始字节核验。接续迁移时先保存新回执，再更新累计映射、必要关系及current指针；工具和生成入口由同一解析结果刷新。禁止修改旧原件／旧哈希让新路径被动通过；指针缺失时仅保留旧文档映射兼容，版本不符则拒绝。

同一记录的预算或决定别名冲突在主表显示`conflict`，展开双方字段、来源与时点；缺失显示`unknown`。不同时间的归档事件仍独立展示。盘点的`access.jsonl`记录实际通过或拒绝的内容读取，不根据扫描阶段是否跳过链接来推断后续行为。
