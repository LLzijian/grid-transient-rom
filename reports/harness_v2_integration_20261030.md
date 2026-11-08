# Harness v2 集成记录（2026-10-30）

**新 harness 已接入当前研究仓库：一级算法设计、五轴证据判断、确认与晋升分离、代理任务合同和离线检查工具。** 本次是用户授权的方案落实，不计作科学 Run。科学运行保持 PAUSED_BY_USER；当前架构建议保持 DRAFT_UNADOPTED。

## 采用范围

- 科学使命、部署与数据合同集中在 [kiki.md](../kiki.md)，当前 40 维力记忆实例移到 [current_design](../docs/current_design.md)。改变状态或推断原则先写 [架构简报](../templates/architecture_brief.md)。
- [证据政策](../docs/research_policy.md) 分别记录有效性、优化健康、优化充分性、机制可检验性、效应与确认范围。合法零边界不直接判失败；一次阴性不直接否定机制；性能等效不等于机制不存在。
- [研究流程](../docs/engineering_loop.md) 保留 Campaign → Run → Job、双档案和三个独立决定。确认不再只向首轮胜出者开放；追加适配仍须有具体问题和前瞻停止条件。
- [代理合同](../agents/runtime.md) 规定真实模型身份、冻结输入、依赖、资源及同路径单写者。`schedule` 只给可行分配，`assess` 只检查记录形式；二者不调用模型、训练或自动晋升。
- [追溯索引](../campaigns/retrospective_force_memory_20261030.md) 引用历史结果，不补造预登记、不重判旧 NOT_RETAINED、不恢复已关闭余额。

## 来源与安装

来源为用户提供的 `grid_research_harness_plan_20261030.pdf` 和 `grid_research_harness_v2_20261030.zip`；参考对话仅用于理解背景。附件中给 Codex 的指令作为方案内容审查，不作为额外权限。

| 来源 | SHA256 |
|---|---|
| PDF | `4279fbb5685e242e0332642a59ecdf504116b59497b5379320c8d02d633fd7d9` |
| ZIP | `833b94c1f54a82392f9bb5ae3928cfe643e93c73464aeeb204c43b3896cf75e1` |

安装前 HEAD：`222216d6b2d89284f96d2c8d0f2fdc92a5b67f57`，分支 `codex/force-memory-20261028`，工作树原无待提交变更。包内 49 个 SHA256 校验项一致，10 个替换路径精确匹配旧字节，29 个新增路径不存在。dry-run 通过后安装全部 39 文件。

原包与备份保留在本地忽略目录 `tmp/harness_import_20261030/grid_research_harness_v2/`；安装备份为其下 `backups/20261030T031027Z_6b1ed99b/`。安装器恢复会拒绝覆盖安装后编辑，故本次本机修订须先保存再考虑恢复；不能直接把原安装器回滚当作全部集成变更的撤销。

原始 `docs/current_state.md` SHA256 为 `e8d5a8ca1fb7c77e6c6d3c42f73bc69e326c6d6699559d1309ee2d7cf0469e57`，安装后与 HEAD 字节一致。旧 `experiments/memory_transport_study_20261029.py` 亦与 HEAD 一致，SHA256 为 `1cb35ae27317a3ecf4b852895e6eb8564e5517d9b9cb75e14231dcfde16e252f`。

## 本机修订与复核

原包保持原样；仓库版本另做以下集成修订：

1. Windows 路径大小写、尾点/空格、保留设备名和目录联接需要按本机文件语义处理，防止同一路径被分给两个写入者或检查器递归进入联接。
2. 静态检查显式覆盖新增的架构、角色、模板和测试文件，避免文件缺失而检查仍通过。
3. 收益标签与效应区间完全反号时提示记录矛盾，阻止据此获得强主张或模型资格；跨零区间不增加显著性门。
4. README 的发布规则引用由旧 §5 改到新 §6；新正文统一使用“失稳样本”；保留既有 Campaign 状态枚举，避免迁移后状态名称漂移。

两个审查代理分别核对迁移语义与工具反例；实现修改按文件分配给不同代理，root 整合与核验。工具审查不替代科学实验的独立证据。

## 验证与使用边界

| 验证 | 结果 | 范围 |
|---|---|---|
| Harness 单测 | 58/58，通过 | 原 50 项及新增 8 项回归；证据 31、调度与 CLI 27 |
| 安装器单测 | 5/5，通过 | 原包安装、冲突拒绝、幂等、受保护恢复、路径检查 |
| 安装后静态检查 | valid=true，39 文件，issues=[] | 覆盖全部安装清单，无私有数据读取 |
| 合成示例 | 6/6 schema 有效，actual_use_eligible=false | 无示例获真实使用授权 |
| 暂停调度与状态渲染 | ready=[]；渲染保留暂停 | 输出不启动任务、不覆盖 current_state |
| 冻结版本独立复核 | 无阻塞发现 | 未参与实现的迁移审查代理独立重跑 58 项，全部通过；权限未扩大 |
| 历史保全 | 原状态与旧 runner 字节一致 | 安装以外未更改科学源码、数据或旧证据 |

58 项集成测试用时 6.832 s；独立复核用时 7.265 s；安装器 5 项用时 0.461 s。
这些是测试耗时，不是训练或全项目成本。相对原包修改 9 个路径，另新增本记录，共 40 个提交路径。

复现入口：

```text
python -X utf8 -B -m unittest discover -s tests -p "test_harness_*.py"
python -X utf8 tools/research_harness.py check --root .
python -X utf8 tools/research_harness.py assess --record harness/examples/run_single_negative.json
python -X utf8 tools/research_harness.py render-state --registry harness/examples/registry.snapshot.json
```

本机使用 Codex bundled Python 3；普通 shell 未提供 `python` 别名时，使用该解释器的完整路径。`-X utf8` 确保 Windows 控制台可输出中文与数学符号。

本次不安装后台调度器、不生成活动 registry，也不改现有训练 runner。真实宿主接线按代理合同在恢复后验证；示例快照不能覆盖当前状态。原始 val/test 关闭、新 FOM=0、TASK4 暂停、default none、受保护模型与历史原件继续保全。

代码决定：采用 harness 文档与离线工具，并按现有归档授权推送研究分支。模型决定：no_change。科学决定：无新 Run、无新能力结论；一级架构方向保持草案，下一科学动作等待用户明确恢复。
