# 本地命令的模块搜索路径

本机 `.venv-gpu` 没有以可编辑包方式安装工作区的 `kiki`。运行本轮命令时显式设置
`PYTHONPATH=E:/03-nn_Transfer_Func/kiki/src;E:/03-nn_Transfer_Func/kiki`，并在工作区根目录执行；
spawn工作进程继承该设置。BLAS、OMP、MKL仍由已审脚本固定为单线程。

首次元数据准备在导入阶段以 `ModuleNotFoundError` 退出，尚未进入 `prepare()`，
没有创建claim或prepare回执，也没有模型或资格调用。具体命令回执见
[环境修正记录](../../results/gftd_matched_prediction_20261109/startup_environment_correction.json)。
此次仅补齐进程环境，不修改冻结源码、协议、比较或预算，首次资格attempt仍为0。

```powershell
$env:PYTHONPATH = 'E:/03-nn_Transfer_Func/kiki/src;E:/03-nn_Transfer_Func/kiki'
.venv-gpu/Scripts/python.exe -B experiments/gftd_matched_prediction_20261109/run.py --prepare results/gftd_matched_prediction_20261109/prepare_release.json
```
