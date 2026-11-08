# 四次记忆计算工具：独立工程集成审查

2026-10-20，独立代理`/root/fixed_edge_final_review`。本次只审可选runtime、13项合成检查、说明及六个依赖的固定main差异。未执行测试/模型/数据/新试验，未读取权重或新科学成绩，未修改绑定源码、旧审查、协议或执行设置，未递归委派。仅新增本文，root保留合并和科学决定。

**支持runtime、对应合成测试、必要短说明三文件的范围受限工程PR。** 六个被报告为字节不同的旧依赖，经本次核查全部仅为行尾差异，不存在源码内容差异；不需要顺手合并它们，也无需为此重跑已关闭的QA。接受依据是实际本地QA和与固定main依赖的源码等价，不能说已在main重新运行QA。

## 六个依赖的确切差异

从GitHub fetch_file逐个读取仓库`LLzijian/grid-transient-rom`、固定ref `3e97af12092545ad727e216e4afb33adb0d405f7`，直接使用structuredContent.content的已解码源码；没有把源码当JSON解析，没有checkout或Git写操作。六文件统一CRLF→LF后逐字符完全相同，逐行比较也没有增删。为排除工具替换行尾的可能，本次还按Git blob格式对远端解码UTF-8字节计算SHA1，六个均等于API返回blob SHA；本地文本UTF-8字节的blob SHA又与本地原始字节独立计算一致，无BOM隐藏差异。

| 文件 | 固定main blob SHA | main字节 | local字节 | local额外CRLF数 |
|---|---|---:|---:|---:|
| experiments/generate_private_dataset.py | 69d11d64b478101ed3800c3535c37c40a00ddc25 | 23478 | 23729 | 251 |
| experiments/run_private_m2.py | c4097123388d222a294ce511159ddf6ce9363e64 | 7162 | 7297 | 135 |
| experiments/train_joint_latent.py | 8692ee80cddb9ad3b31577f79d339c8fd23ebd54 | 28488 | 28955 | 467 |
| src/kiki/data/private_dataset.py | 6e0415efc6c23031888e506184037ea1447728b8 | 3362 | 3427 | 65 |
| src/kiki/models/private_pm.py | d7d17926fbd9582252eecb48103e5c197ea185ec | 4231 | 4323 | 92 |
| src/kiki/models/private_rom.py | 75d2271e533c5953d56c2810eb828ff837ae058c | 3923 | 4014 | 91 |

Python源码读取对这些行尾进行统一处理，所以差异不改变导入定义、类、函数或数值语句；本次没有需要用“无法触达”掩盖的语义差异。合成runtime测试从toy_source构造模型，使用内存合成事件，检查原目标、Adam、通用场、NumPy、导出和临时Adam文件，不调用这些依赖中的真实数据生成/读取入口。该路径没有依赖上述六个源码文件的原始字节hash作为数值输入；相关数据身份摘要属于各自数据/研究入口，不能把不同原始字节冒充同一科学绑定。本审查不修改277源身份或将科学绑定改为规范化hash。

root已报告完整38个旧文件依赖闭包中另外32文件字节相同；本次独立针对剩余六文件核查，不将root的闭包报告称为本代理重新执行的全仓检查。精确PR仍须只选择三个合格文件，保护local用户/历史改动。

## 实际QA与工具范围

当前runtime SHA256为53af6d143c66ba86b1f0cb503e5a791bf604218acf8d388e7ed098fb583c1a68，测试为b402eed5ae23bbc652a1d6d20b9f72047ec8da8cb52ae6590ceba32f5c31dfea，与上一实施审查及实际QA所用版本相同。

`results/quartic_memory_20261020/qa01_result.json`记录源码commit `31f100d39402ae7d78fd8c32c9525dc974d596f2`，原生24.865688499994576秒，两层退出0、277源前后一致、worker78272及创建身份、清理GONE，实际Torch2.13.0+cu130/RTX4060Ti。公开stdout逐项记录runtime13、runner8、native9，合计30通过、0失败。13项runtime包含CPU/CUDA非零四次能量/场/独立NumPy与有限差分、原点梯度/Hessian、零a公共raw梯度与非锁死、同参数集合fast/generic裁剪Adam、只投影a与矩保留、term重排/分组、双副本与负值拒绝、非空Adam下一更新、单次初始化以及cache/非有限错误清理。

这些检查足以支持保存可选计算工具：四次项、原目标共同计算及投影优化有实际正确性/恢复依据。完整30项不是全都属于拟公开runtime测试；runner/native检查使用本轮私有监督路径，不在三文件PR内，不能宣传为公开clone自足CI。本次没有因依赖疑问添加第二次QA或改断言，先前唯一前置纠错和原记录继续保留。

## 说明与合并条件

`docs/quartic_memory_runtime.md`对q潜坐标、a的系数身份、图缓存、有限性、投影保矩、导出双副本及非线性稳定边界的描述符合实现。建议在QA句中补明“上述13项为固定本地环境、源码31f100d…的实际结果；与本PR基准main的六个旧依赖差异仅为行尾，未在main重新运行QA”，并将完整30项标为含实验runner/native的本地工程QA。root可只改这段未绑定的说明，不改运行中源码；不必创建新测试链。

三文件PR不得带入上述六个旧依赖、实验runner、私有launcher、源checkpoint或新默认模型。合并前由root核基准main、三文件精确blob及最终说明，之后按既有expected-head流程操作。若main在核对后出现新的实质依赖变化，应暂停合并核变化，不能把本意见外推。

本工程接受不证明四次结构改善预测、提高失稳安全、加快计算或完成机制发现。当前科学轮的prepare/cost/训练/评价仍受原协议、精确字节绑定和门槛约束；本意见没有新执行授权，不改变默认、P4或val/test边界。

本次单次审查最多600支持秒嵌既定3800hold，不新增原生任务或预算，不重复计费。本文写毕停止编辑。
