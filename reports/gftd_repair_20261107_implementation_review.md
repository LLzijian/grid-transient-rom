# GFTD 5N 核心与观测探针实现审查（2026-11-07）

结论为 `eligible_for_stage0_qualification`。本次有界源码审查未发现阻断登记 Stage 0 的实现问题。探针为 `method_consistent_pending_stage0_and_release`，root 核验正式资格结果、源码哈希与单次执行身份后，可释放八基准局部探针。该判断不授予模型晋升或结构恢复结论。

冻结输入的 SHA256 如下。

```text
design       7E378CA751F3B2C9AB7B610B0698FE10929EEC94B7CDDC74B1736F4F47B95E45
functional   2155837548DDB196DF5F65D727000F091E59411EA816941076A21A27FDC27452
model        D951805CEF802A952899A4D10922BCEE73A20BC6D0A9C8EC78B75DAA0DB8DD1F
benchmark    9A87B64A70D2B4A7E5D4DCE60F98B263A044A163201B5E0C10DB4CF3373493EC
core_tests   E4B69B496201E1A423DF03AA95FDFBCAD6DD56739DA1DBFFCC43557B3E83B4AA
probe        B8E7E1AA96AFBE0E4757D799002310C1939D2E10191F07C03FBDF603CACAABCC
probe_tests  0A350CC573EC37839525FD3E1F7FD4024B2D5BF3905E773747D42255D67B9DBB
parent       ECD5F4A7980B748EEB7DA78986733C42324C44BEA73D0CDA964BFB62B1269D1F
parent_H     921B6DEDB2E8B5218C5F4EC84FC6A35AC4844EBC0E310F8A6639BB0C7BCF88F3
```

[functional](../src/kiki/physics/kuramoto_functional.py) 将独立 p 放入储能和全部梯度，RHS 同时包含 +G S_p p 与 −G omega。幅值和力记忆分开，指数核与真实导数一致，浮点越界显式失败。COI 动力门与记忆 Laplacian 明确拒绝，半正定性没有被写成全系统解耦。本轮实现限于对角耗散。

[父映射及存取](../src/kiki/models/gftd_kuramoto.py) 保留有效父参数，将边映到完全图并对重复边累加，新增边权精确零。j_mem=kappa、lambda_mem=1/kappa、tau_mem=1/(kappa lambda) 给出 −r 与 kappa omega−lambda r。边反向同时翻转相角与参考差，中心势保持一致。初始化回到 theta−delta_star 调用父编码器，仅填末点 p₀。检查点保存全部 functional 字段、门、编码器权重、父配置与完整父状态，加载拒绝旧 4N 格式。fit 与非零调制初始化明确未实现。

[探针](../experiments/gftd_observability_20261107.py) 实际参数给出 kappa=0.7、Q=0.3I、R_a=0.2I、S_p=G=I、R_p=0.3I，与冻结八基准一致。硬端点消去的尺度、二阶前向 s 导数、合法符号、历史 0.0025/0.00125 s 与未来 0.01/0.005 s 对齐。已检查全部 s=0 基准的幅值零列及同步平衡的 s 零列，输出辅助块三阈值秩、近零子空间投影、有限端点 p 漂移和所选方向响应。RMS 字段是人工尺度下混合相对角度/频率差异，p 漂移为尺度化二范数，均未冒充物理单位 RMSE。

源码审查与实际资格通过分别判断。正式测试与 probe 未由本 Reviewer 重跑，当前运行结果为 unknown。末点有限漂移和谱/子空间数值稳定性须随结果判读，`COMPLETED` 只表示执行完成。已知范围边界是父初始化器加载依赖 experiments 在 Python 导入路径内，本轮仓库测试有该路径，独立安装包的父检查点便携加载尚未验收。同族 benchmark 未作为独立 Stuart–Landau 证据，也未进入八基准探针的数据来源。
