# 资格导入路径修复复查：接受重新准备与首次有效固定资格

**接受这次单行导入路径修复。第一次资格尝试在测试收集阶段停止，回执没有任何模型调用；旧失败及字节身份完整保留。可以重新生成元数据准备，再执行第一次实际进入测试体的固定资格。** 原128次RHS／2000次H总上限不变，真实25包仍须等待有效资格结果和 root 阶段释放。

日期：2026-11-09。Reviewer：`/root/feasibility_reviewer`。本复查只读源码、归档和JSON回执，没有执行测试、模型求值、优化、ODE或读取私有观测。原[实现审查](implementation_review.md)保持原字节，SHA-256 `25a7791449f55b613634a0bb878dabaf7dacb3ce1938e3314c9e2dc10b8156d3`；其公共实现、执行流程、预算和科学范围判断继续有效。

| 本次绑定 | SHA-256 |
|---|---|
| 修订 `experiments/gftd_bounded_initialization_20261109/qualify.py` | `e6c7ad176fac9718b9597de113c1b932c0387dca9292dc3bcd8a13e3d4971a63` |
| [首次尝试驱动原字节](qualification_attempt01/qualify.py) | `29c07d6fdf924095011aa980ca8d738f86782b249f2fc6e8b0bdbc7982c28b3f` |
| [首次资格回执](../../results/gftd_bounded_initialization_20261109/qualification_attempt01/qualification.json) | `44b290bdd12cdc6e5f6ddda0d656f0cd569cbfd094dc942951a3cd77480e023f` |
| [字节归档映射](../../results/gftd_bounded_initialization_20261109/qualification_attempt01/byte_archive.json) | `3bb63480fb5e54956a26a3506c12f7f067ee6b1406e74b1658ec93bf5703b4bd` |

归档映射中的六项均已核验SHA；其中五项声明的字节数也逐项相符，驱动原字节与原实现审查绑定的SHA一致。归档包括首次prepare、prepare claim、qualification、qualification claim、候选checkpoint及旧驱动，没有用成功记录覆盖失败。首次prepare SHA为 `37b849d04ccc0f946f3145c4627990c550e7fb62fc18b2aa6d93bff5ba4fdb5e`，首次候选checkpoint SHA为 `f8eabacfeac00074f135c7af826d8d89d8f23c445c788485d0fb79a0f54b0637`。

**失败与零模型工作。** root原执行回执 `8bde47` 报告测试收集时导入 `experiments` 失败；持久化资格记录为passed=false、pytest_exit_code=2、counts={}、violations=[]、new_ode=0及new_initialization_optimization=0。已审驱动在调用pytest前安装计数profile和禁止真实积分／优化的包装，收集失败没有进入固定测试体。结合该调用路径，RHS、H、梯度、字典及所登记物理函数的实际调用均为0；这不是因字段缺失而把未知执行改写为0。首次失败仍计入技术修复历史，耗时wall 1.2019359999976587 s、CPU 0.90625 s保留。

**变更范围。** 与归档驱动的逐行差异只有在ROOT定义后增加 `sys.path.insert(0, str(ROOT))`。该路径是由驱动自身位置确定的工作区根目录，使测试收集可以导入本轮 `experiments` 模块；没有更改测试选择、fixture、源哈希验证、预算、计数器、禁止调用包装或模型方程。原资格回执绑定的全部源码逐项核对，仅该驱动哈希改变；公共核、字典、初始化器、模型存取、runner及两个测试文件保持原审查版本。

**后续接受条件。** 本次状态为 `import_path_repair_accepted_for_reprepare_and_fixed_qualification`。重新prepare只重建候选与来源元数据，不读取私有观测内容、不运行模型；新prepare必须绑定上述新驱动SHA。随后允许一次固定资格，沿用原正常／饱和点、两个静态对照、同点克隆和mock范围。第一次尝试消耗模型预算为0，后续实际调用仍与原总上限共同核算，不能把新目录或新claim解释为新科学额度。预期14次RHS／374次H等仍是源码预测，须由有效资格回执实测。

真实pilot与full的放行条件保持原协议。未复跑资格或模型；剩余待核查项是修复后测试能否完成、实际计数是否对应固定范围，以及新prepare／资格／候选身份是否一致。
