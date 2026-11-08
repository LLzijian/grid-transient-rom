# P8工程接受与科学关闭

root接受独立终审限定范围：仅可选P8 runtime、对应runtime测试、说明，全部来自已实际运行的21项通过子集。整轮42项的41通过/1失败原分类保持，失败runner/native及相关测试只归档；无QA02、成本、训练、评价或模型采用。

[PR21](https://github.com/LLzijian/grid-transient-rom/pull/21)精确三文件，base1e4d51c497a77fa7da3a507d8fc007e27631a3aa，expected head2458d0edaf0676584cdd26550ad5b940a23ed36a。合并后再次GET PR确认merged=true、merge924965604883177b8603e4010d06905b6c0a797f，并核main同值和三blob：

- runtime：b048a4adf674c25ace9ac8735571f7f2640b4cc7。
- runtime测试：d50a4be8a6b577bc38587835731ba7e18bbc67f6。
- docs：16975dbb87d69a65831452f603566d933421134f。

静态import闭包50个已有依赖44原字节匹配、6仅换行匹配，无缺失/不符。冻结私有合成资产另需提供，不声称main自足CI，也未在合并后再次运行QA。测试未触及真实权重或train/val/test数据。本地HEAD/index再次断言保持，没有checkout/reset/stage/强推或删除证据。

实际费用及hold见results/cubic_port_only_20261021/closed_cost.json。报告关闭持久化首个命令因百分号格式串失败、第二个修正成功，2.0954986+1.918818秒属于已held的归档支持1000子集，未额外新增科学费用；不能据该写作故障复活本轮。P8执行关闭不等于父任务完成，后续必须先真实Astra规划不同的合法有限问题。
