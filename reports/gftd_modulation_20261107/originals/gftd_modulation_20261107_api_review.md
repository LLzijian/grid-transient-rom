# Run2 数值基础与 API 独立审查

结论：冻结数值基础的方程、尺度和接口主体与正式协议一致；API 有一项需修复的存取边界。此报告仅覆盖基础模块和 API，不接受尚在实现的拟合引擎，也不释放科学执行。

## 绑定输入

| 文件 | SHA256 |
|---|---|
| `src/kiki/models/gftd_discovery.py` | `9245B0726CB3FCA15D3248106B43BD1673ED6607ED3A852B67C533F768A1B27F` |
| `tests/test_gftd_discovery.py` | `232F16DF69C4E0932E3D585359F37CD4E5FC9ECCF9B852F6F0D33F0D6B6764C0` |
| `src/kiki/models/base.py` | `F374726BCA448EC73C8A138E41EDE55939B0137830DFD038C13530DF3EA8039D` |
| `src/kiki/models/gftd_kuramoto.py` | `C39EE8D694E1005B5EB1B32AEDB160D8DD5D0321C39B0574943BC59A755B6539` |
| `tests/test_gftd_kuramoto_architecture.py` | `6EB6ED2FDAA1FC9B57ECA6B7528871BF43B54EE76D508512011C3B3A030473F0` |
| `tests/test_gftd_model_fit.py` | `0119D3FFAC4AE4D176BA26504B25F3C766B8D2501517B0643B69CA25608B3481` |

正式协议为 `54622AD9DA5FEA1AFB326140C6D4288A51BDDCA2AAE896EF10926F951B09DCB0`。本次只读源码与测试，没有读取变化中的引擎或 runner，没有运行拟合、积分或新采样。

## 必要修复

`gftd_kuramoto.py:393–397` 在成功提交新 `causal_two_stage` 模型时保留 `parent_initializer`。因此 `from_parent(...).fit(...)` 的存档仍在 `536–540` 行保存已不用的旧父初始化器；`load` 在 `569–579` 行、判定初始化分支之前，无条件导入旧 `r_memory_model_20261025` 并重建该对象。新因果模型会额外依赖旧实验模块。普通新构造模型的 fit 路径不受此问题影响。

最小修复：在新 fit 的原子提交处清除该旧引用，保留 `legacy_static` 的原行为；加入父模型转新 fit 后的存取回归。修复前，此存取路径为待完成，不能把现有普通构造器 round-trip 测试外推到该路径。

## 已核对的合同

- 基础模块按完整 p/r 端口重建，保留实际 gain、storage、memory 门和衰减系数；弱式符号、0.1 秒尺度及恢复物理 r 的尺度一致。Boole 求积、四种弱测试和固定物理缓存检查保留。
- 条件字典在给定候选 a 时取相角偏导，精确保留指数核；实际非负系数为 `[k−β,β]`，解释字典为 `[A_static,A_exp−A_static]`，二者没有混作同一个信息判据。相对 KKT、空问题计数和两遍 MGS 的 Q 构造与协议一致。
- fit 严格接收五键、拒绝非空 val、要求同 N/σ，复制输入和功能体后调用引擎，验证返回配置后才提交；`best_validation_loss=None`。initialize 为四键，σ来自存档配置，要求返回有限 5N 状态及逐值相等的真实末点 p，类型化失败保存诊断。旧 v1/from_parent 的初始化与推进分支保留，新因果分支缺配置拒绝。

代码决定：保留基础/API 实现，修复上述单项后复核。模型决定：不晋升。科学决定：允许继续既定引擎实现准备。父方报告的 26 项基础、37 项 API/mock 与父回归通过属于工程资格；本次未重跑，也不将 mock 引擎输出当作实际引擎、初始化数值资格或科学证据。完整预算、选择规则、停止条件与真值隔离留待全源冻结审查。

## 单项修复复核：API 接受

修订源码 SHA256 为 `395EA4D7CA50AEB07247049CB96AD0BDB9A884EC141F3C7591EBC780331FC901`，`tests/test_gftd_model_fit.py` 为 `2A0B11956E54C2A2F4D90F58887A39EBDA83978BECF7FDAC2B1C6F6696411658`。逐字节备份位于 `results/gftd_modulation_20261107/prelaunch_api_v1/`，两项旧哈希与本报告绑定值相符；base 与 architecture test 的当前字节与备份相同。

独立 diff 确认生产代码只在成功 fit 的原子提交处新增 `self.parent_initializer = None`，没有改变旧 `legacy_static` 分支。新增回归先用真实父 fixture 验证旧初始化，再经 mock fit 保存 `parent_initializer=null`，禁用旧模块后仍能加载，并验证新因果初始化一致。该变化直接消除了本报告指出的无用依赖，未扩大信息或接口。

结论：原 API 问题已闭合，接受本次基础/API 阶段实现。父方报告修订后 38 项 API/mock 与父回归通过（pytest 9.56 秒、exit 0）；本次未重跑。此接受不覆盖实际拟合引擎、正式样例或真实初始化资格，不释放科学 action；完整实现仍须待全源冻结独立审查。报告至此停写。
