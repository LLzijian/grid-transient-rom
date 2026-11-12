# 当前研究状态

**2026-11-13：OSGKD 阶段一算法与合同切换完成。GFTD 旧支线已暂存，未执行的线性补偿界工作量已关闭；本次未启动科学计算或阶段二训练。**

活动架构：[OSGKD](../designs/osgkd_architecture.md)。活动问题：[IEEE39 三端联合辨识](../campaigns/osgkd_ieee39/campaign.md)。主合同是 motion_only_v1：11 点/100 ms 相角与完整转速历史，加声明的冻结常数；真实末点 p₀ 只保留为 known_p0_reference 的信息增强输入。

机械侧学习驱动作用及动态，耦合侧学习有效连接与有界调制，环境侧表达有来源的静态作用及必要记忆。训练仅使用 fit297 中已允许且实际存在的端口记录与未来轨迹作监督；部署不读取它们。固定参考、单位、覆盖与分解锚定由后续 adapter 核对，本次未读取私有科学数组。

## 当前指针与已收束工作

- 研究分支 `codex/osgkd-transition-20261113` 从已核对科研 HEAD `9dec0976` 派生，main 未改动；阶段一结论见[采用回执](../results/osgkd_transition_20261113/adoption.json)、[核查结果](../results/osgkd_transition_20261113/validation.json)和[独立审查](../reports/osgkd_transition_20261113/review.md)。
- OSGKD 尚无训练模型或已释放的科学 Run。原 quarter 工作候选降为 GFTD 诊断档案；default none、incumbent 和 protected anchors 不变。
- 线性界脚本两版保留，第二版复查未完成，资格与科学均未启动。执行保留 PAUSED_BY_USER，后续动作 defer、auto_dispatch=false，预算 closed，科学 unresolved。[收束证据](../results/osgkd_transition_20261113/closure.json)
- 已完成的匹配预测、三点历史比较及局部补偿结果不重写。[一页 GFTD 总结](../reports/osgkd_transition_20261113/gftd_closure.md)
- 核对时未发现本项目 Python 3.12/虚拟环境计算进程；四个其他 Python 进程未操作。没有新建自动推进目标或周期任务。

## 唯一下一动作

本次交付阶段一算法与合同。后续阶段二首先将既有端口数据映射与完整 C0/C1 接口一并实现，再按有限协议做训练—评价—解释；不把旧补偿界、完整唯一性或 L2 证书设为前置。本次请求只完成第一步，附件的自动进入阶段二指令未被采用。

原 val/test 关闭，旧消费 test 永久退出选择，新 FOM=0，TASK4 暂停；不扩大到 train385，不增加部署输入或解除受保护模型约束。
