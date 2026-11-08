# 独立工程结果与精确PR审查

独立审查者 `/root/fom_history_interface`，非实现者。结果/源码审查4.3838992秒，精确说明/证据字节审查1.7046546秒，均计已登记结果/PR180，不重复收费。未改文件、读取checkpoint内容、运行数值或递归。

原始结果审查实质返回：

结果/工程范围审查：允许将本轮已验证core+tests作为可复用工程成果，配准确docs+capability_profile选择性main；不构成科学性能/模型采用。源为1aa93d60de8199455ef563acfa3e83f9ffc04eb8，当前module SHA8d0b19eaeefe168944b8937f207526f3a0ced058b49f51b0262d962c26b31177、tests SHA95ec213c56958d9c3ecb4b907115f0afdd670391fedfbb67e7425fdf1596122a，与actual QA/cost immutable map逐项一致。

已核：8 QA实际通过（stdout30.72s/native52.6231316），含两形式CPU/CUDA梯度、真实32→34模型/Adam/RNG/输出精确恢复、缺step/空groups负例，以及两形式独立NumPy RK4完整50步逐步比较；非零tanh与旧实现含raw梯度/clip/Adam参数映射。严格Adam检查已含groups/ID映射、step/exp_avg/exp_avg_sq/shape/finite及观测到梯度的参数状态。cost_only_trace用真实32 loss+padding至262144，load前后真实读/哈希/比loss32；该具体cost路径不是单独QA用例，但在唯一完整cost全部12组合实际通过，不能写成另一个QA项。

完整cost实际计数384更新、12/12生成、12完整评价、48保存、12恢复、24构造、12加载、36raw、256JSON齐全，native354.7782015；两层0/GONE/source一致。T_eachrun3547.5814484，T_eval77.46901，T_total14267.7948038，总25%+350=18184.7435047>12000；各run及总预算门槛失败、eval门槛通过，REJECT正确。保留所有慢样本/首步费用，不缩规模/重测；正式12×512与正式val均为0。不能据此判断有理优于tanh、语法能力不足或模型族失败。

待写文档必须明示：仅工程检查与完整费用试算完成；384次cost训练和timing数据确实执行，避免说“未训练任何模型”；CPU/CUDA梯度QA采用短工程目标，50步是独立导出推演，不误称完整0.5s训练梯度。公开4文件依赖已有main模块、实际CUDA/Python/GPU环境，完整原生CLI复现还需固定归档中的私有P0/P1 helpers/bindings/协议/回执和本地checkpoint路径SHA，不称四文件独立CI或已授权运行。原val/test/TASK4/default保持。docs/profile尚未写，未对未见字节作最终精确head批准；按这些条件写即可，无需新增数值检查。

最终精确字节补充返回：

补充实际字节审查接受：PR28 exact head4536a53210a7f37334b4d1300e853ddc8d2cb7ff、base59fc3046e959bb1c8717accaeb744c1315597bd4，可按四新增文件的工程/证据范围合入。doc SHA256=7d08a43480e1b8be16f4661c4cac03ebe931e63e4be13530f615d9f1cbfefddf；profile SHA256=d61db402d141ef0b3a49b85e85affa55c4e39eeddc650684d5081b7605dbed27。与先前已核code/test SHA及实际QA/cost一致：8工程检查、50步独立推演、384实际timing更新明确区别formal_updates=0/512端点0；18184.7435047>12000及REJECT陈述准确，没有模型族失败/未知发现/科学采用主张。Windows/GPU环境、原路径恢复私有helpers/96源证明、152本地检查点相关文件和非一键复现依赖已明确披露；原val/test/TASK4/default不变。无阻断项，不授权未来训练/预算。仅一次只读shell native1.7046546秒，计既有结果/PR180；未改文件或数值执行。

root采纳工程限定范围。附加清单核验：152相关文件中150为实际32步源检查点/指针，2为伪512目录的故意负例（元数据实际32），负例明确标注，不作为正式源或512证据。main紧凑证据仅报告总相关文件152，无模型权重导出或科学端点。该标注不改变原QA/费用/PR字节。
