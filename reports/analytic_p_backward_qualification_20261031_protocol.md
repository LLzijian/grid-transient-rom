# 解析反推数值候选的完整资格

Run：analytic_p_backward_qualification_20261031；目的probe。
状态DESIGN_FROZEN，root已采用Scientist设计；真实执行仍待实现与独立审查。
日期2026-10-31（Asia/Shanghai）；Campaign force_memory_closure_20261030。
授权沿用户要求按harness自动迭代；本次由heartbeat kiki接续，不是新的数据/部署授权。
父p_reverse_precision_20261031远端归档已核验be39cfb5f84c079753d3bd250efc94e8a6177ed5，
实际私有archival_verified.json与父源码/输入/结果均须绑定。

## 唯一规范与root采用理由

规范设计：[analytic_p_backward_qualification_20261031_scientist.md](analytic_p_backward_qualification_20261031_scientist.md)，
SHA256 7e174327e77a8364fde0dab4892cd15a094c375169f7d963918e6ca466a9e496。
其§1–§8全部作为本协议的资产、数学/因果/序列化、指标扰动、身份/T、停止、确认与三决定规则。
root选择理由：父probe已解释反推数值误差与初值传递，本轮检验该数值实现是否保持完整自主行为，
以实际资格证据决定能否用于后续工作，而不由小初值差直接推断长期无影响。

## 有限执行与不变边界

仅A96_analytic与A16_analytic两个候选，显式analytic_p_backward_overlay_v1新身份。
先由未改原严格loader加载/核验父模型，再由专用loader核数学来源并只替换p_backward。
原模型/loader/源方程/权重/40维状态/100 ms合法信息均不改，不将候选伪装旧格式。
load-export-reload、早期p占位无影响、缓存和末点硬钉、实际调用及原反向1e−12门全部核对。

数据仅已消费train fit297=267稳定+30失稳；原val/test关闭，新FOM=0，TASK4暂停。
optimizer=0、共享参数拟合=0。正常最多两份新完整paired评价，各候选一次，
future .01/.005 s、history .0025/.00125 s、CPU float64单线程，沿sgm原评分/分类/数值函数。
当前先仅释放正常两份；纯实现修复至多一次，若须重算已生成预测，先对账与记录理由后由root单独放行，
全Run新完整paired评价总上限4份。不因科学/数值门失败改变阈值、事件、步长或候选。

原版完整缓存/原身份集合及各自T_base须在候选读出前冻结；基线无效不自动重算。
数值行为保持门与原科学准入门/T_anchor另表：不通过后者的A16不能因前者通过而晋升。
旧反推门失败、旧M3unknown和上游tm单位限制原样保留，本轮不重判p形式。

## 执行与归档

工具源码/测试/协议/设计及完整输入manifest在执行前冻结，候选及结果不可覆盖。
包装记录放private_data/analytic_p_backward_qualification_20261031_execution，
不要在工具运行前占用其不可覆盖结果目录。原生进程身份只留私有。
完整候选/预测/逐事件明细留private_data/analytic_p_backward_qualification_20261031/，
公开聚合与哈希写results同名目录。未知Job先对账，不盲目重放。
结束后独立审查、三个决定、更新Campaign/current_state/harness，选择性提交并核验远端后再开下一Run。
