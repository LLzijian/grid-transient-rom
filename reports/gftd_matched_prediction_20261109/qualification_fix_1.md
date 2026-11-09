# 第一次资格复查：单位算例的浮点断言

首次固定资格得到43项通过、1项失败。失败发生在预置算例的共同转速RMSE断言：
理论值为2 rad/s，实际float64结果为1.9999999999999998 rad/s，两者相差约2.22e-16。
原测试使用逐位相等比较，未容纳10节点质量平均和开方的舍入。

本次唯一源码修改是该测试断言，改为绝对误差不超过1e-14、相对容差为0。
预置输入、理论值、评分公式、单位、所有科学数值门和原297分母规则全部保留。
该容差用于固定单位算例的舍入核对，不进入模型选优或L1合格判定。

原测试已逐字节保存于
[首次资格源码](qualification_attempt00/tests/test_gftd_matched_runner.py)，
SHA-256 `9aafbf2ec10e60cb66b6f8c3ec87297da766f3a8d56c3b6ad7c9501b4a786288`。
原[prepare](../../results/gftd_matched_prediction_20261109/prepare.json)和
[资格回执](../../results/gftd_matched_prediction_20261109/qualification.json)保持原字节，
后者SHA-256为 `9a13b3baf832bb17138d6d72e9dda9c0e407db06470d04952e1188123f0e09be`。
原执行工具回执为`6d5f79`，exit1，pytest记录26.37 s；资格宿主wall为29.1480605 s、CPU为10.171875 s。

首次实际调用为14次RHS、374次H、14次梯度；模型入口观察数与调用账本一致，
违规为空、cost_unknown=false，没有ODE、真实优化或原始数据读取。
首次资格失败消耗的调用计入原128次RHS和2000次H总额，不清零。
拟使用技术复查1（至多2次中的第一次），重新执行同一固定套件；没有新增物理点或科学试验。
