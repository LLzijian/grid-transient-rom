# FOM与GPU环境分工：工程验收

2026-10-20。root验收已登记的环境接线检查：真实ANDES 2.0.0环境通过生产系统身份核对，Windows spawn子进程身份一致，原CUDA环境确认RTX 4060 Ti可用；9项保护检查全部通过。源码来源eeae1563f734c0ad3023892200725a492ae9047f，90个明确source/config执行前后相同。[紧凑证据](../results/fom_environment_repair_20261020/qa.json)保留真实环境、退出、清理及费用。

FOM解释器为E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe，GPU解释器为本项目.venv-gpu/Scripts/python.exe。两者实际Python均3.12.14；ANDES 2.0.0、NumPy 2.5.2、h5py 3.16.0，GPU为Torch 2.13.0+cu130。case SHA256为9c2048dc94201ee48ffe816de65d53831fa1f4b0e7fdf32f24c50f1017edcfe5；train公共物理指纹为4da835555c89e36b4bb8f0a8e5f7edfce94b6207a38e028f57f34ded46de7b72。没有安装依赖或修改外部项目。

实际worker85932、创建FILETIME134369146707911838；内外exit0，原生清理完成且worker为GONE。正确FOM、正确GPU、错误角色三个真实探测分别exit0/0/1，均通过监督核验。FOM spawn子进程60076真实exit0、身份匹配。错误解释器、prefix、缺包、子身份、源码、case、空身份与不可用GPU等保护在条件callback前拒绝。进程清理复用已验收的原生监督。

唯一QA外层17.84752110001864进程秒，包含所有探测子进程，不另加。加600秒保守支持，结项617.8475211000186；取消未用582.1524788999814回repair。父累计57943.4684632，余额201256.5315368；repair余12174.8522211，P3保持88361.7061617，其余类别与旧900hold不变。无重验。

复现入口为private_data/closure_mechanism_20261019/repair/fom_environment01/launch_qa.py，GPU解释器运行，参数`--source-commit eeae1563f734c0ad3023892200725a492ae9047f --attempt qa01 --cap 450`；实际命令JSON、入口源码和绑定摘要见本结果目录及reproduction子目录。私有原生完整回执为同目录native_qa01/execution.json，公开证据给出路径与SHA256。复现需要本地两个环境和许可的train公共物理元数据；现有输出不得覆盖或重复执行。

本次没有物理求解、轨迹或模型权重读取、更新、val/test访问。工程验收支持后续生产批次调用同一preflight，不代表FOM数值合格、模型晋升或L2证书。原48条件96次缺ANDES失败与原档关闭保持。后续同条件首次物理执行须另外保存实际Astra6方案、root窄政策、完整科学预算与独立挑战。可复用工程待有限PR审查与远端合并核验，默认模型保持。

独立[实际验收审查](fom_environment_repair_20261020_acceptance_review.md)支持有限工程集成；root核对28文件精确PR范围后，PR14以预期head069973d02067745588175c9bd791e301a931bf29合并，远端main核实为6b17be3535cc57985c1e1be4138decabc7c0f033。工程合并不改变上述科学边界或默认模型。
