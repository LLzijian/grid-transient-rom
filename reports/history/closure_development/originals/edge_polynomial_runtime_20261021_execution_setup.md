# 多项式执行后端首次QA来源锁定

root已读完整runtime、专用测试、执行/计时模块及P0原生助手，并采纳独立静态挑战必要项。尚无本轮Torch/QA/probe/benchmark或正式数据执行。只做一个静态共享多项式路线，历史源码/旧池/teacher不改。

- experiments/edge_polynomial_runtime_20261021.py SHA256 64ecce03ed2e1aef7a4c24854298a3197b071dd9e1ef7b5cef9d2ad98805b537
- tests/test_edge_polynomial_runtime_20261021.py SHA256 fc0bacbf6d408ad86e4b7b7006796306033056715cde9e2246d3960c94855f57
- experiments/edge_polynomial_study_20261021.py SHA256 e5d4f010fa9263f9c8e66b7ba3a2a000b88607fefd9a50b61772e9527e1e85fe
- tests/test_edge_polynomial_study_20261021.py SHA256 c49c28b218fb58a57bfc4413c1fc9d269763fb96f66fadbf61d890be7ae4d8d0
- private_data/closure_mechanism_20261019/P0/edge_polynomial_runtime01/make_invocation.py SHA256 1b7e2b97cd10a38ef5e49e06e300b560e2a56d7c9cda705ff429e818110e8539
- private_data/closure_mechanism_20261019/P0/edge_polynomial_runtime01/launch_stage.py SHA256 3c21f49e89542b5adc5a913b48a7dc5a1431fae7e954cf67e1bcd9c890747de8

准入数学容差和工程收益/完整T≤576均按协议锁定，不根据结果改门槛。全六项共同参数/Adam/RNG/批次重置后配对计时，预热和首次完整操作计科学max，工程中位数取后3次；固定I/O各类次数乘实测max，实际合计另报；最终评价实际测六项、正式计3*max。raw写盘含SHA。三次冷探针native记录保持，root后填真实generation+complete-launch外部wall凭证（不含虚构时间），其全外部费用按P0元数据一次计，内部native不叠加。部分benchmark结果逐候选保存。

源码实现保留物理c、端点对称、单位、J/R/port方向；稀疏使用轴和边方向仅图内共享，with_terms/load/to/结束清缓存。新base工厂RNG/flags/参数顺序保留，转换后建立Adam；旧teacher独立生成。测试额外覆盖非单位惯性/功率尺度。主QA650/635/620，唯一明确实现缺陷修复完整关键QA300/285/270，必须显式root声明源码差异/失败回执/新源；成本开始后不再修复或重试。

独立方案6.9969235、实现7.8179685工具秒归P2review180，原始Astra约15.31归既有P2规划300；runtime叶14.5385082、study/helper叶16.4466731工具秒归P0support1500，root读源/修正/记录/归档等也归相应支持；均不假称精确总实耗。主QA、3冷探针、唯一benchmark和条件P1待真实收费。新P0≤3600/条件P1≤900/P2review180的类别和旧未知hold保持。当前无数值作业。
