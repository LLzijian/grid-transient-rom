# 私有数据与过渡期保护资产

新数据集放在本目录；实验模型、预测、缓存和运行记录放在 `artifacts/<实验名>/`。
原始 val/test 的关闭状态及原划分不因目录整理改变。

主要数据集为 `ieee39_postfault`、`ieee39_expansion`、`ieee39_holdout_supplement`、
`ieee39_boundary_0075`、`ieee39_boundary_fine_0075`；其原文件和划分保持原样。

其余暂留目录见[保护清单](../results/private_data_cleanup_20261107/protection.json)：
它们包含当前研究读取链、唯一仿真数据或历史重要模型。部分旧代码将解析后的绝对路径
纳入身份校验，因此待阶段边界验证路径兼容后再迁移。权限受限而未核清的测试目录也保留，
不把无法读取当作空目录或可删除依据。

已经迁出的文件按[旧→新路径清单](../results/private_data_cleanup_20261107/migration_map.csv)
查找；历史协议和原始回执不批量改写。历史重放若依赖旧布局，应按迁移清单恢复独立副本，
不得在当前研究输入位置覆盖重跑。

完整执行结果和空间统计见[2026-11-08清理记录](../reports/private_data_cleanup_20261108.md)。
