# N/Q1024事实提取准备

本工具固定共同385、原366、新19身份和六个N/Q1024角色，仅复用已测试空间提取器的纯JSON函数，不调用其硬编码O/S主入口。纯函数源码SHA、cohort SHA、新1024日程SHA、父O512来源manifest SHA均固定；当前轮冻结manifest SHA由root在全部六评价完成后显式传入。

实际调用先检查全部六complete/evidence SHA、队列角色及来源provenance，再读取指标和JSON训练记录。没有缺角色补行、重新推理或优化。原8 s与实际15 s资格独立；任何相关事件未决使对应总体值为null，保留分母。公共结果包含1/2/5/8 s稳定风险、原COI/频率/功率指标、失稳真实区间及分类、空间解释和局部39维事实。原事件与逐步训练追踪只写本地。

a按AST身份定位，检查terms/coefficients/state_dict双副本。stage0及每32步JSON按可训练参数名、AST和标量位置对齐，计算N/Q公共参数更新范数与差异，新增a不参与公共向量比较。记录投影、raw/clipped gradient、clip事实及513步学习率变化，不读取Adam二进制或重放更新。该JSON证据不能单独证明优化充分或Adam矩正确。

新增每个h的实际次数及其前64/后64次更新的loss和原组件均值；样本随日程变化，不据此宣称固定样本收敛。a的raw gradient（clip前）、gradient（clip后）、投影前后和更新量保存有限范围、正负零与缺失次数，并分前512/后512及h报告。精确N7/Q8 AST集合在所有保存节点验证。local.json必须引用冻结端点model SHA，完整标记及local文件的SHA一并存档；不匹配即未决。

协议保护项输出全精度比较布尔值和未决值，包括三seed不增及至少两seed含6101漏检减少的事实；最终科学决定、物理/来源/功率检查和候选采用仍由root负责。当前没有读取任何正式新训练/评价成绩，未执行真实提取。合成测试完全不依赖私有资产；执行费用单独返回root计入提取额度，不消耗生产模型QA修复名额。

等待root核准后执行：

```powershell
& PYTHON private_data/closure_mechanism_20261019/P2/quartic_memory01/postprocess/extract_quartic_memory.py --after-complete --manifest-sha256 FROZEN_SHA --output NEW_JSON
```

只有`--after-complete`不能替代六份完成、hash和身份检查。输出文件已存在即拒绝；发生异常保存UNRESOLVED，不产生部分总体成绩。
