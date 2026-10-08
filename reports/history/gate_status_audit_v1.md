# 技术报告:仓库迁移到封闭式门控工程协议(基线审计 v1)

> **历史归档 · 2026-09-22**：历史过程记录：早期门控建设已完成，本文中的待办和工作树状态不再有效。
> 原文保留供追溯。有效入口：[0001 验收](../accepted/0001_pruning_horizons_and_module_adoption.md)；[当前状态](../../docs/current_state.md)。

日期: 2026-09-05 · 状态: **完成(审计)** —— 项目于 09:24 被 git init 并提交**基线**
`f764fe0 chore: establish audited repository baseline`(`main`,工作树干净)。此后
`docs/engineering_loop.md` + `configs/acceptance_gates.json` + `reports/accepted/0000_repository_baseline.md`
成为本仓库的**治理规则**。本报告把仓库对其**自声明必选门**的运行状态如实审计一遍。

## 1. 已建立的基线事实
- 仓库根 `E:/03-nn_Transfer_Func/kiki`,分支 `main`,HEAD = `f764fe0`,**186 文件已跟踪、工作树干净**(满足 loop "start from clean worktree" 前提)。
- 提交内容:README、协议文档、门注册表、configs、全部 experiments/run_e*、paper(源+bib+3 图+main.pdf)、src、results(2.6 MB/100 文件,不可变证据入库)。LaTeX 中间物与缓存已被 `.gitignore` 排除;`main.pdf` 作为审阅产物保留(符合 .gitignore 注释意图)。
- `reports/accepted/0000_repository_baseline.md` **已随提交入库**(基线验收记录)。
- `.claude/`(个人会话配置)未入库(0 跟踪),状态良好。

## 2. 对自声明必选门的运行审计(用 kron-reduce venv,不新建环境)
`acceptance_gates.json` required = G1-syntax / G2-unit / G6-json。

| 门 | 命令(映射到 venv) | 实测 | 结论 |
|---|---|---|---|
| G1-syntax | `compileall -q src experiments tests` | exit 0 | ✅ **PASS** |
| G2-unit | `pytest -q` | "no tests ran";**collected_tests = 0** | ❌ **NOT PASS**(注册表判据要求 `collected_tests > 0`) |
| G6-json | `verify_artifacts.py --json-only` | **文件不存在**(exit 2) | ❌ **CANNOT RUN**(缺 `experiments/verify_artifacts.py`) |

## 3. 缺口与成因
脚手架留下了两处必选门无法满足的基础设施空缺:
1. `tests/` **为空** → G2 永远收集 0 个测试,按其注册表判据必败。
2. `experiments/verify_artifacts.py` **缺失** → G6 无命令可跑(文件未创建)。

另注:门命令写的是裸 `python`,需在本机解析到项目 venv 解释器;工程上应把门命令显式
绑定 venv(属后续小改,不在本审计执行)。

## 4. 建议的下一"一致变更"(单 loop,遵循协议 1-实现→2 干净→3 单变更→4 跑门→5 接受/回退)
> **Register the claim**:"让必选门(G2/G6)可运行并通过" ——
> (a) 写 `experiments/verify_artifacts.py`,实现 `--json-only`:递归解析 `results/`+`configs/`
> 全部 JSON,校验 well-formedness + 关键 schema 键存在(含 main.pdf/manuscript 数字可回溯性
> 的轻量钩子);(b) 补最小 `tests/`(确定性契约/单元测试 >0,如数据契约字段白名单、
> 能量模型 nonnegativity/供给恒等式等只读既有代码的断言,不改科学结果)。
> 预期效果:跑通 G1+G2+G6,达成"仓库能对其自身必选门自证"。反证/回退:任一必选门失败即
> 回退并留 `reports/failed/`。

## 5. 边界(沿 0000 文档)
本审计**不验收继承的科学主张**(轨迹精度/模块选择/零漏报等仍由既有审计管辖,须经后续
claim-specific 门(如 G5)单独通过后方可"promote")。本报告只处理**仓库治理层**现状。
工作树干净,下一 loop 可直接开始。

## 6. 待确认
是否由本 loop 直接实现 §4 的 (a)+(b) 并跑门提交,取决于你对该变更范围的授权(涉及新增
源文件与一次 commit)。给出确认或调整即执行;否则维持当前干净基线待命。

---
## 更新(2026-09-05,§4 已执行且必选门全绿)—— v2 addendum

**并发设置方在此期间亦在推进**:新增 `tests/test_verify_artifacts.py`(定义
`verify_artifacts.py` 的**期望 API**:`inspect_json_text`/`verify_json_artifacts`)、
`tests/test_module_adoption.py`、`tests/test_pruning_horizons.py`、`pytest.ini`、
`results/README.md`,并有 `reports/accepted/0001_pruning_horizons_and_module_adoption.md`
已写就;`run_e2.py`/`run_e6.py`/`src/kiki/models/prunable_energy_model.py` 被其修改(未提交)。

据此,本 loop 完成 §4 的(a)+(b),且**与并发方测试规约对齐**:
- `experiments/verify_artifacts.py`(已实现,依 `test_verify_artifacts.py` 规约):
  `inspect_json_text(text, *, source, strict_nonfinite)`(重复键→error;非有限数
  strict→error/legacy→warning;畸形→error)+ `verify_json_artifacts()`(results/+configs/
  全扫,legacy 模式,只以 error 判 fail)+ CLI `--json-only`。
- `tests/conftest.py`(src on path)+ `tests/test_causal_contract.py`(6 条确定性契约断言:
  forbidden∩allowed=∅、因果静态场合法、label/valid_length/fault 禁入、prefix_steps<2 抛错)。

**必选门实测(全绿):**
| 门 | 结果 |
|---|---|
| G1 syntax(compileall src experiments tests) | ✅ exit 0 |
| G2 unit(pytest) | ✅ **14 passed**(6+2+3+3;含并发方 3 个测试文件) |
| G6 json(verify_artifacts --json-only) | ✅ scanned 74、errors 0、status pass;warnings=92(= 语料中合法的 Infinity,如被拒模块 `time_constant=inf`,legacy 记 warning 不判败) |

**边界与处置:**
- 仓库扫描用 legacy 模式是**有意为之**:继承语料含合法 Infinity,strict 会误杀全库;
  严格 non-finite 留给新建手写文件场景(strict_nonfinite=True 分支已备)。
- 本审计完成时,工作树含本 loop 与并发设置方两批未提交文件,因此本审计没有代替
  科学循环提交。随后主循环独立审阅这些变更,通过 14 项测试及全部注册门,并以
  `085df48` 接受。本文作为该提交前门控状态的历史记录保留,不构成额外科学结论。
