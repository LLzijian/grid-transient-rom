# 语法外因果工具：主分支精确范围审查

结论：批准以下精确四文件作为已验证的工程工具及费用证据合入main。它们提供了可复用的有理端口teacher、因果自主学习入口、真实训练状态冻结/恢复检查和完整费用测量，不构成预测能力、结构发现或模型采用结果。无需为本次集成再执行数值任务。

候选head：`693524a5341d531ff44ba863137d952a4b76a230`；base：`7419d196fe1d080221db01565835f424c7a47f2c`；分支：`codex/outside-grammar-causal-tools-20261021`。root已核远端比较仅这四项新增；本审查独立读取本地四项并计算SHA256及Git blob，与root给出的候选blob完全一致，未自行执行Git或远端操作。

| 文件 | SHA256 | Git blob |
|---|---|---|
| experiments/outside_grammar_causal_20261021.py | 8b90b53a3479d9a297feb7a78a0bfe9f6da4cad9577b8ce6ce218c2388c9cf0d | f155aba349cb001bd68d516aacbeb8143a392616 |
| tests/test_outside_grammar_causal_20261021.py | ed5a9af63c6db917721a2c05b10605c85df5e17b52550ddae254d407001cd2b5 | 843f5a3403767a050ada6eedf8c61183e10a20bc |
| docs/outside_grammar_causal.md | 5d6e82183ba0c524fce4aa1c35dd8c938173ca626ffc979dfab2f5f4325f4086 | 26df8233bfebaa13e55eaef15f73c2e4d30cb361 |
| results/outside_grammar_causal_20261021/capability_profile.json | d0b40d66ef6c5b1fd01db8e80e64a1350853f3530d3f2a3310d68dbd92a916b0 | 5c1f8113b503b6a96d8bcf5840d2fbe5de49c496 |

## 接受依据

冻结数值源码为 `e761605f2d328dc3be97458863c5df0671eceea9`。已保存QA摘要记录9项通过、native34.1494695秒；真实32→34恢复与伪512负例均在检查范围内。此前指出的四项实质修正已在候选中落实：生成计时含资格检查；冻结读取实际计数、trace、身份和非空Adam；最终加载计时包含train.npz解包/拷贝及两次端点检查；正式和cost共用advance逐步梯度记录。真实训练状态验证在正式val生成之前执行，不能凭目录名或哈希把32步伪称512步。

唯一cost完整退出，native76.6776534秒；96更新、12保存、3恢复、6构造、3加载、3组train/val生成与完整评价、9原始写盘和64原子JSON均有完整计数。准入记录T=1422.2936666秒，run含余量1755.4606100秒、evaluate含余量22.4064733秒，总含25%余量及180支持为1957.8670833秒，超过900帽；总门槛及run门槛失败，evaluate门槛通过。文档和capability_profile均诚实标记REJECT，并明确正式科学端点为0、正式数据流未执行。费用失败没有被改写为预测失败或表达能力不足。

QA与cost摘要均记录同源、双层退出0、清理完成及GONE。文档把工程检查与未启动的三seed×512能力表分开，没有沿用试算成绩、宣称未知公式恢复或修改默认模型。主分支集成是保留有实际正确性/恢复证据的通用工程能力，原科学模型资格保持不变。

## 复现范围与合入边界

文档给出实际CUDA/Python/GPU/float64环境，并链接固定归档 `cb04439bf744e6560d9a6c5f6242cd5d4276be08` 的完整设计、回执及费用证据。原始数组、真实Adam checkpoint和Windows监督助手仍按登记路径/来源保留，说明没有将这四文件包装成独立CI即可完整复现或可直接启动未准入研究。原项目val/test关闭、TASK4暂停、默认模型不变的表述与本次四文件范围一致。

批准仅对应上列base/head与四个blob；不批准额外P1助手、正式训练、默认模型更换、预算转配或未来研究。root继续负责实际PR范围核验、合并及远端提交确认。

## 操作记录

本次只读四文件及紧凑QA/cost/准入摘要，并计算文件身份；只写本报告。未运行Torch、测试、模型数据、推理、Git变更或递归代理。原生工具精确wall秒数由完成消息给出，计同一outside P2review250。
