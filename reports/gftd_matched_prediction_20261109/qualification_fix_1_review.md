# 第 1 次资格技术复查审查

结论：接受此次单行测试修正，可按冻结协议进行技术复查 1。首次失败及其全部实际成本必须保留并累计；本决定不释放科学拟合、部署或真实评分。

本次只读资格回执、prepare/claim、原测试归档、当前测试和修正说明，核对字节差异并作 AST 解析。未运行测试、模型、ODE、优化或数据计算。

| 证据 | SHA-256 |
|---|---|
| 首次资格 `qualification.json` | `9a13b3baf832bb17138d6d72e9dda9c0e407db06470d04952e1188123f0e09be` |
| 首次 prepare | `fe6e0677341a7e63bc25f1e6e01077feb812492c294d52a63aae84dfa0de58b5` |
| 首次测试归档 | `9aafbf2ec10e60cb66b6f8c3ec87297da766f3a8d56c3b6ad7c9501b4a786288` |
| 修正后测试 | `d51d5310b63b227c10e32122de3b447f6239fe73adbc5a2acc03ec7ec51dc284` |
| `qualification_fix_1.md` | `5c3903c2b304c7271ad44fcd493659e99f4b72a21548f2deedb2ff67d22b89a5` |
| runner v2 独立审查 | `1ff4d90dcfd0424c08f8f062614b1d2ecc0418631a1dba8bf810c984145a149c` |

上述实际字节哈希相符，资格绑定的 prepare 与 claim 也相符。对首次冻结的 43 个源码文件逐项核对，唯一变化是 `tests/test_gftd_matched_runner.py`。当前不存在科学拟合阶段的 started claim。

root 执行记录为 43 项通过、1 项失败，native `6d5f79`、exit 1。机器资格回执为 `passed=false`、`pytest_exit_code=1`、`error_type=null`，违规为空且 `cost_unknown=false`；实际入口观察与独立调用账本一致：14 RHS、374 H，另有 14 次直接梯度、8 次字典调用。记录中的 runner Radau 入口属于已审 mock 测试，不能据此计为真实 ODE。首次资格未通过，不能改写成通过或零成本。

差异仅将固定共同转速 RMSE 算例的 `==2.` 改为 `pytest.approx(2., rel=0., abs=1e-14)`。理论值仍为 2 rad/s，执行值 1.9999999999999998 的差约为 2.22e-16，符合质量平均与开方的 float64 舍入量级。该绝对容差只用于人工单位断言；预置数组、评分函数、单位、主细数值门、LOS 合同、原297分母及官方结果资格均未改变。没有新增物理点，也没有借失败修改科学选择标准。

技术复查 1 是至多两次明确缺陷复查中的第一次，继续同一固定套件。新 prepare/release 应绑定本说明、修正说明及新的测试哈希，`qualification_attempt=1`，`previous` 引用上述原资格回执。原总上限仍为 128 RHS／2000 H；已用 14／374 不清零，当前余额为 114／1626。若再次失败，继续保留失败、源身份和实际成本；未知成本不能自动复查。

只有复查实际通过、累计账本和源码身份闭合后，root 才能另按登记阶段释放科学工作。本审查确认技术复查条件成立，不预判其测试结果。
