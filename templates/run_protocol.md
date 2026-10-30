# Run 协议 <run_id>

Campaign：<campaign_id>；父 Run/候选：<id>；目的类型：probe / adaptation /
discrimination / confirmation；状态：proposed / registered / running / analyzing / closed
来源身份：<commit + checkpoint/model hash>；信息范围：<data/cohort>
设计入口：kiki.md §<n>；机制卡：templates/mechanism_card.md（<mechanism_id>）

## 问题
<本轮要检验 Campaign 的哪个关键部分；它减少哪个重要不确定性；若结果为负会改变哪个
结构性判断>

## 设计
| 项 | 内容 |
|---|---|
| 臂/对照 |  |
| 固定项 | 状态维数、信息、因果初始化、评价口径、数值资格 |
| 改变项 | 只列本轮真正改变的机制/参数化 |
| 数据与分母 | train 事件集、真实区间、不可初始化/未决处理 |
| 指标 | L1 精度/关键事件/时机/成本；L2 若适用 |
| 数值资格 | 步长/精度/等价性/梯度 |

## 优化与探针
<优化充分性判据、检查点、延长触发；L2 探针与预期签名（见机制卡）>

## 快速迭代（Run 内）
<允许的小循环：改实现/适配—短训练—比较—保留或丢弃；边界：不改数据/信息/评价口径，
不用固定短时间或单指标贪心替代机制验证；每次试错计入预算并入实验树>

## 停止与决定
<有限预算；何种结果继续/转向/关闭；确认条件；三个决定如何记录>

## 风险与替代解释
<混淆、冗余参数化、可辨识性、对照不足；不可行的救回方式>
