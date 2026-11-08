# 第五批：历史主题、顶层收拢与结果关联

本批核查536项材料，归档210份，326项保留原位；累计归档416份原件，内容与换行字节保持不变。另为更新前的结果首页保存一份字节副本。

## 阅读入口

- [后续闭合模型与训练时域](../../../reports/history/closure_development/README.md)
- [力记忆与p通道](../../../reports/history/force_memory/README.md)
- [执行与恢复维护](../../../reports/maintenance/execution_runtime/README.md)
- [仓库与运行规范](../../../reports/maintenance/repository_runtime/README.md)
- [全部材料与路径例外](../../../reports/history/catalog.md)

本批同时接入GFTD的5份旧草案、准备及专项审查原件，最终冻结协议保持原处。旧保护项中2项因引用组已收拢而解除；其他重新分类不算解除路径依赖。

## 结果与本地产物如何查

[关联表CSV](storage_associations.csv)覆盖96个公开结果目录、216个旧本地迁移根和27个保护／受限根；[完整JSON](storage_associations.json)保留每条公开引用、原字段、旧／新位置、登记上下文、已知材料角色和实际记录的执行标识。

21个正式Run均可从关联表追到相关结果或来源。目录被某份报告引用，不自动表示该报告产生了目录内全部文件；来源Job无法确认时保留null，不按名称补造。

关联表是可维护的阅读视图，后续代码归档可更新其中的当前来源路径，原始来源路径始终保留；冻结科学回执和采用决定不回写。最近的[路径更新回执](../20261109_code06/association_path_updates.json)接续9个旧入口的归档。

6个本地保护根尚未找到公开引用线索，分别为三个 `p_constant_direction_EQ_engineering_20261031*`、`task2_long_0060_2s_eval`、`task2_structure_0078_extend_C-free_eval` 和 `task2_structure_0078_extend_G-fixed_eval`。它们保持原路径与原保留决定，不能由缺少引用推导为可删除。

本地观察只检查根路径元数据，没有遍历或打开原始数据、模型、预测和缓存，也没有重算大文件哈希。旧文件数／字节数及旧哈希核验仍引用原迁移回执，根路径存在不证明所有子文件可访问或可恢复。

## 核查与撤回

- [预检](preflight.json)、[实际迁移与累计位置](migration.json)、[验收结果](validation.json)。旧回执不覆盖。
- [材料清单](catalog.csv)、[全部保护线索](protected.csv)、[非Markdown待核查项](pending_non_markdown.json)。
- [原文链接解析](outgoing_links.csv)：742份原件的固定提交内容已核对为相同或仅换行不同；720份Markdown中的486个内部链接分别解析。两个公式被普通Markdown正则误识别，已按原文逐项标明，不算缺失文件。

撤回时按迁移回执逐项核对新位置的大小与SHA256，确认旧路径未被占用后搬回；允许维护的导航可从预检前镜像恢复。本批没有移动科学代码、配置、数据集或本地产物，没有重新训练、预测或改变采用决定。外网链接可达性未检查。
