# TASK4：独立研究项目

与现有 `kiki` 并列放置。目标：100 ms历史初始化、机器级预测记忆、功率结构内学习、再压缩为显式ODE。详见TASK4.md。

```text
工作目录/
├── kiki/                  # 已有项目：完全只读
└── TASK4/                 # 本项目：代码/派生数据/模型/报告写在这里
    ├── AGENTS.md
    ├── START_HERE.md
    ├── TASK4.md
    ├── agents/
    ├── configs/
    ├── docs/
    ├── scripts/
    ├── src/task4/
    ├── tests/
    ├── state/
    ├── reports/
    └── artifacts/
```

## 先读这三个文件

- `START_HERE.md`：把启动语句交给Codex/Claude Code等执行代理。
- `AGENTS.md`：自主权限、接受/回退与方向审批。
- `docs/implementation_status.md`：已实现组件和需要代理继续实施的研究阶段。

**常规规划内工作自主推进；当代理认定核心方案有问题并准备改换方向，必须暂停询问，明确认可后才继续。**未使用预算返回本项目父池，不作废，也不回填旧TASK2。

## 环境

Python 3.11+。在TASK4新建环境，不写KIKI环境。

Windows PowerShell：
```powershell
.\scripts\bootstrap.ps1
```

Linux/macOS：
```bash
bash scripts/bootstrap.sh
```

安装脚本会创建环境、安装声明依赖、跑单元测试、检查源路径；**不启动真实训练**。依赖安装需要网络，torch的GPU构建应由执行代理按实际硬件处理，不私自升级用户全局环境。

不安装为包也可运行脚本（仍需依赖）：
```bash
python -m pytest -q
python scripts/preflight.py
python scripts/smoke.py --output artifacts/my_synthetic_smoke
python scripts/project.py status
```

合成烟测不需要kiki，不产生电网性能结论。输出目录不可覆盖旧运行。

## 数据路径

默认`../kiki`，不是`./kiki`。可修改configs/project.json或设置TASK4_KIKI_ROOT。只读两套HDF5的TRAIN/validation名单及M3基线；不会复制数据或读取TEST载荷。数据缺失时给出准确路径并保持BLOCKED_DATA。

## 正式执行入口

推荐让执行代理按START_HERE完成授权登记、实验规划和命令调用，不手工跳过独立审核。

状态工具示例（`instruction`必须引用真实用户消息，不得伪造）：
```bash
python scripts/project.py authorize --instruction "实际收到的用户启动与预算指令" --cap-seconds 60000
python scripts/project.py open-loop --id T4_0001 --spec reports/T4_0001_plan.md --reserve-seconds 20000
python scripts/project.py run --cap-seconds 7200 -- python scripts/train.py --config configs/experiment.json --arm P --output artifacts/T4_0001/P
```

这里20000/7200是命令示例，不是强制科学预算；实际登记依据测时，预留最终自主精调和评价。当前starter训练命令只产出T4-N候选，不会自动完成整条符号压缩路线或判断科学接受。

## 回退与改向

代码快照：
```bash
python scripts/snapshot.py create --output artifacts/code_snapshots/candidate_001.zip
python scripts/snapshot.py restore --snapshot artifacts/code_snapshots/BOOTSTRAP.zip --expected-current artifacts/code_snapshots/candidate_001.zip
```

工作树若已被用户额外修改，restore会拒绝；不能强行丢弃。模型回退使用活动指针，不删除失败模型。

方向变更：
```bash
python scripts/project.py propose-direction --proposal proposals/DIRECTION_001.md
# 此时停下询问。只有得到针对该提案的明确认可，代理才能记录：
python scripts/project.py approve-direction --instruction "用户实际的认可原话"
```

`state/research.json`记录状态，不替用户审批。程序不会联系外部模型、不自发运行、不保证宿主关闭后仍在后台研究。

## 证据边界

打包时完成合成工程检查；没有真实kiki/private_data，因此没有电网训练、性能提升或稳定域证据。文件夹完整，研究仍需执行。请勿把BUILD_VALIDATION中的合成结果当作TASK4科研结果。
