# 早期 E 系列：冻结执行入口

这里保存9个已闭合的早期入口与诊断原件，迁移前后SHA256一致。[研究认识与纠正链](../../../reports/history/early_e_series/README.md)和[迁移回执](../../../results/repository_organization/20261109_code06/migration.json)分别说明科学范围与文件位置。

这些脚本按旧路径深度计算ROOT、设置导入或定位输入；归档位置用于保存与查阅，不直接运行。需要恢复旧执行时，使用下表的完整原提交，在独立检出中恢复原目录和对应环境／私有输入，并重新核对原数据权限。不要将当前源码、当前数据与旧输出混作一次复现。

| 原入口 | 原版本阅读 |
|---|---|
| check_data.py | [数据检查](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/check_data.py) |
| profile_e2.py | [E2计时入口](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/profile_e2.py) |
| run_e1_m0.py | [E1骨架](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e1_m0.py) |
| run_e2.py | [E2模块](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e2.py) |
| run_e3.py | [E3机械功率](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e3.py) |
| run_e3b.py | [E3b分析](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e3b.py) |
| run_e4.py | [E4能量筛查](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e4.py) |
| run_e6.py | [E6旧确认入口](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e6.py) |
| run_e7.py | [E7基线](https://github.com/LLzijian/grid-transient-rom/blob/145847532bacd3454dca8869e819778efc8258f6/experiments/run_e7.py) |

原val/test的关闭与已消费历史继续有效。恢复源码不恢复数据或重跑授权；旧科学主张沿后续审计解释。
