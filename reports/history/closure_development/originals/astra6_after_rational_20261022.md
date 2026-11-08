# Astra6规划与root采纳

真实调用：collaboration.spawn_agent(task_name="astra6_after_rational_20261022", model="gpt-6-astra", fork_turns="none")，返回task_name /root/astra6_after_rational_20261022；模型覆盖来自实际调用参数，不以自称替代。完整原始返回见astra6_after_rational_20261022_raw.txt。只读，无训练/评价/状态加载/文件修改。

root采纳：下一有限科学问题选既有六候选生成语法中的结构选择。已知有理形式收益已建立本轮受控结果，但未知表达式选择缺少实际运行证据；先执行曾因成本未准入而未运行的完整三seed六项研究。保持旧语法/teacher/seed/更新/排名规则，不扩展有理运算，不重复已看的合成val，不直跳真实val/test。

开跑前核旧QA/cost未产生正式6101/6102/6103数据，冻结选择旧正式命名空间；若历史污染则必须在本次执行前另统一登记，不能运行中选。root保留科学门槛与结论；一次相关QA集合(可复用未改旧7项真实证明)、最多一次实现修复/相关复验，不重跑全历史QA或成本。新入口复用旧search_seed/科学函数，checkpoint中不添加checkpoint_*诊断文件；新增冻结核验真实step/Adam/trace/来源。窗口train拟合0.5s、排名(2,5]，三冻结后才一次新合成val。

恢复范围在协议前瞻明确：不在首次薄入口中开发任意外部--resume；发生真实中断即保存关闭，之后另登记非择优来源续接，原最多一次界限不重置。内部32/128接续沿用真实共享serializer。取消进程额度不增加科学分支，有限2304正式更新，无择优重跑。
