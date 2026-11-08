# fit297 L1 NumPy适配合同

范围：train预期297=267稳定+30失稳；仅源码核查。根E:/03-nn_Transfer_Func/kiki，文件均在experiments/（P除外）。

1. 输入与对齐
合法初始化仅11×10 theta/omega、11点[-.1,0]、末点真实p0；sigma0。初始化只做一次，主.01/细.005从同一实际返回50维点出发，不分别初始化。完整输出主(1501,50)、细(3001,50)，含t=0；8秒切片801/1601点。
score50：前10列theta−delta_star，其余保持[omega,p,a,r]；score30=score50[:,:30]。pred[1]对x[11]，初点对x[10]；LOS加回delta_star一次。
event需id/group/label/x(T,30+)、classification_meta（equilibrium_delta/inertia、omega_s、event_time_s）。mass/p_scale正10维，绑定冻结hash；分类元数据不进ODE。

2. 误差与聚合
J:363、C:382定义e=pred−truth，b=mass/sum(mass)，P e=e−1(b'e)。固定sδ=.47675314588683515，sω=1.0033667249602924。
逐点q=.5 mean((P eδ/sδ)^2)+.5 mean((P eω/sω)^2)+(b'eω/sω)^2+mean((ep/p_scale)^2)。
先各事件时间平均q，再用K:174：risk=.8 mean+.2 mean(最大ceil(.1N)个事件值)，不是RMSE，也不是按时间点加权。不用H:203 weighted_risk；分项risk之和不等于总risk。
B:735提供稳定完整1/2/5/8秒risk，点数100/200/500/800；primary为G:151稳定完整5秒risk。C:407失稳实际区间n=min(1500,T−11)，每事件等权；不能用A的8秒prefix冒充实际15秒。
F:50归一化COI RMSE：SSE=.5(sum(Peδ)^2/sδ²+sum(Peω)^2/sω²)，n=10×步数；返回delta_sse/omega_sse/normalized_sse/n/rmse。O:50聚合返回mean/pooled/cvar10/worst/worst_group_mean及物理delta/omega RMSE。
共同omega误差=b'eω，单位rad/s；p归一化误差=ep/p_scale，维数无量纲。A:28/38、C:493池化采用sqrt(sum SSE/sum n)：共同omega n=时间点数，p n=10×时间点数，不用事件RMSE均值代替。N:15另有物理p指标（100MVA）。
细[::2]正式评分，主格同范围另算，存两格risk/RMSE及差。

3. 数值资格与LOS
D:75逐1/2/5/8秒及prefix检查所有原始主/细子步、全部50维有限；COI归一化主细RMSE与共同omega归一化RMSE均<=.001。比较包含t=0至范围末点。A的prefix最多800点；真实实际区间用H:216/255，最多1500点。
D:51 LOS-aware比较：两格均触发时到较晚trigger，触发时差<=.02秒；均不触发比较全窗；仅一格触发不合格。latency_uncovered保留专类；细格检查每个内部子步，不只抽样点。G:146合并8秒pair及available scored行；B:735的5/8秒risk要求此资格，1/2秒用自身资格。实际区间资格用I:658，同时要求actual interval与pair有限、available/admissible、los_pair_pass及两类差均<=.001。
E:102/123/157/174：theta恢复原角，omega_pu=omega/omega_s+1，默认omega_s=2π60。LOS为角跨度>π且惯量投影相对速度max> .02pu，连续5个.01秒采样，trigger为第5点；尾窗1秒角跨度<π、相对速度<.005pu为window_settled。
分类字段category/predicted_los/window_settled/trigger_index/finite_before_hit/first_nonfinite_index；未触发另分未settled/数值未决/延迟未覆盖。event_time<=.10+1e−9为延迟未覆盖。历史前10点+预测(含初始化点)拼接后分类，不重复历史末点。
E:237时间：记录坐标trigger×.01，决策坐标再减.10；绝对坐标加clearing_time及record_start_delay。时机误差=预测−真实，仅同范围有trigger时可比；8秒和实际区间分报，不计记录终点后命中。

4. 最小接线与失败分母
复用D._metric_safe/_scored_numerics/_pair_row、E上述LOS/共同omega纯函数、G.physical_metrics/I.actual15_qualified；禁止高层evaluate/run、val读取、_rollouts/encode及futurePm。
q/risk及H.interval15含Torch模型接口，NumPy适配仅按上述公式转写；旧A.event_metrics固定8秒且带Torch调用，不能直接充当50state纯NumPy入口。显式检查预测长度，不静默截短。
逐事件/范围保留available/finite/numerically_admissible、n、SSE/RMSE/q均值、错误、主细pair和两套分类。A:47规则：任一eligible失败则该范围official=null，finite-only仅诊断。初始化异常保留id和297原分母，不能移入unavailable以删除；稳定267/失稳30、缺记录时长、初始化失败、数值未决、延迟未覆盖、漏检/误报各列计数。未settled无LOS不称稳定；a/r参加有限性检查。

SHA256：
A summarize_train_local_memory_0098.py
468fc24781df3ce8f184226f125add7becf1e8b573ae074fc89846c1fe99e7a1
B r_memory_study_20261025.py
a6d96ce1ad8a6b006ab5e6201d23c21e8d611836128fa3a77f7d17da8f81bd8b
C closure_probe_20261030.py
f4a3f2f766ebe02fdf2dd8ecaa3a13884628612e7f452b48976eedc7564eb78c
D evaluate_structured_rom.py
38aacbf79e61081449adcb254d99f0556f65021d2bd5b8d46cf17c5d757e01d9
E evaluate_l1_latent.py
5882dc6814f13ef4210d94db07fd87ca88d8130aa984dbda7757c53088662314
F evaluate_long_latent.py
8a52a2c8138d768cd636a1aef8a07fd871b86ed62f78e8785e0d069066ac890c
G fixed_edge_pair_20261019.py
6f3194387f13a103a04bd73d27dbe17cde7e786d8617655ba2fff8bf811d72c3
H active_fom_pair_fit_20261020.py
bfc74e84b8aced9588618b60dd6440adcbf96c1dc6610ea60001425eec623720
I frequency_dissipation_20261021.py
f83edeedf24c40078dd92e2bdaaf3af9739b838027b4179167341218d6188c7f
J train_structured_rom.py
6afa2186411a0a23ea5f5d62f9cfeacb3192c6023f62676c7c134ef9bb365482
K train_local_memory_0098.py
e73de06b9a1d66b05de4bb1d29504a19cc7b486bc3694ddd341d451f403ae667
N closure_discovery_confirmation.py
43ecf8f1b29231e1b68b452efa7310fcdce305a136e4a5bba54e0fbedff66579
O run_private_m2.py
25698fe4fa77b35982b85d1fa84b5e2ec0c9ab728d095e951b8ac0fe9479f972
P src/kiki/physics/coordinates.py:6
9a738d46647ab631407f8e4b5814fe0a7868c717e31fda13963861c10e373089
