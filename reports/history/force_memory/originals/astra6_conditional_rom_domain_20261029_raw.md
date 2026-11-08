# 固定参数ROM的条件性解析内域

2026-10-29。实际模型gpt-6-astra。本次只读了当前force_graph_model_20261028.py的field、energy_and_gradient、matrices，以及state_gated_memory_model_20261028.py的effective_weights、field、matrices。未读实际参数、事件或作业，未做数值条件检查、覆盖率或初态误差计算。本文为纯符号准备，不改变正在运行的SGM，不判定任何端点已获证、L1通过或满足原完整L2要求。

## 条件命题

root提出的rho常数成立，有一项定义应明确：D和gamma_min只对**正势能边**E_U={e:k_e>0}取约束/最小值；不要求零势能边满足相位条件。若另外希望约束其他边，须另有正的边界裕量，不能直接沿用本证明。

固定gamma交换项为0，采用当前同一ROM的全部固定参数，假定：

1. M和S_p为正定对角矩阵；kappa>0、lambda>0且有限。机械耗散R_p=diag(rpp)严格正定，D_omega=diag(damping)半正定。源码A_p=R_p S_p，在当前对角结构下rpp逐分量严格正与A_p逐分量严格正等价；一般非对称矩阵下不能只以A_p“正”替代S_p R_p S_p正定。
2. 所有k_e>=0，E_U构成连通图；对这些边|phi_e|<pi/2。phi_e为当前固定工作点的同一展开约定，不允许为通过条件而单独挪动边相位。
3. Q(q)对称、局部Lipschitz且Q(q)>=lambda I。当前A/C的非负边及余弦门满足此结构条件；B若固定sum(a)>0则同样成立。固定全零图可数学上定义Q=lambda I，但不能据此绕过B训练时的零图梯度/控制退化分类。本文不改变该分类。
4. 无外部输入、参数固定、原状态/时间/尺度定义不变。

这些是充分条件，不声称必要，也不声称当前模型满足。

## 39维商空间及能量

设P=I-11^T/10，q=P delta，q属于1正交子空间。只消去共同角度，不消去共同转速；状态(q,omega,p,r)维数为9+10+10+10=39。运动学必须写作q_dot=P omega，不是omega，也不是把omega换成9维相对转速。因b_e^T1=0，U和Q只依赖q，与共同角度无关。

记G=diag(jpi_p)，则商系统为

    q_dot=P omega
    M omega_dot=-grad U(q)-D_omega omega+G S_p p-r
    p_dot=-G omega-R_p S_p p
    r_dot=kappa omega-Q(q)r.

U(q)=sum_e k_e[cos(phi_e)-cos(phi_e+b_e^Tq)-sin(phi_e)b_e^Tq]。

U(0)=0，grad U(0)=0，grad U(q)属于1正交子空间。因此grad U^T P omega=grad U^T omega，投影不破坏能量抵消。保留全omega非常关键：r_dot=kappa omega仍约束共同转速。

    H=.5 omega^T M omega+.5 p^T S_p p+||r||²/(2kappa)+U(q)
    H_dot=-omega^T D_omega omega
          -p^T S_p R_p S_p p-r^T Q(q)r/kappa <=0.

这里只对H沿向量场求导，H中没有Q，因此无需Q_dot。弱式消元对Q乘以观测导数的项作分部积分，才产生Q_dot；两者不矛盾。H是该ROM的储能函数，不宣称真实SI能量；M omega只是内部动量坐标/惯性加权转速，kappa保持等效刚度类单位，不称惯量。

## 势能下界与边界常数

对e属于E_U定义

    gamma_e=(pi/2-|phi_e|)/2>0
    D={q属于1正交子空间: |b_e^T q|<gamma_e，对所有e属于E_U}
    w_e=k_e cos(|phi_e|+gamma_e)>0
    L_w=sum_e w_e b_e b_e^T
    mu=lambda_2(L_w)>0，gamma_min=min_e gamma_e.

D为包含0的凸开集。在闭包D上，|phi_e+b_e^Tq|<=|phi_e|+gamma_e<pi/2，故

    Hess U(q)=sum_e k_e cos(phi_e+b_e^Tq)b_e b_e^T >=L_w.

沿0到q的线段积分，得到U(q)>=mu||q||²/2。这里mu是欧氏零和角坐标下的谱隙，不能直接混用其他加权坐标的特征值。

连通性使闭包D在零和子空间有界：任意两节点间选有限路径，其角差由路径gamma之和约束，再由零和条件约束各分量。因此闭包D紧。边界至少有一条正势能边满足|b_e^Tq|=gamma_e。对源码标准incidence，||b_e||=sqrt(2)，于是

    ||q||>=gamma_e/sqrt(2)>=gamma_min/sqrt(2)
    U(q)>=mu gamma_min²/4，q属于边界D.

所以选择rho=mu gamma_min²/8，即严格低于上述边界下界的一半，是合法保守内域阈值；不需要修正为更小常数。

## 紧致、正定、不变与前向存在

定义

    K={q属于闭包D，H(q,omega,p,r)<=rho}.

H在闭包D上正定：U提供q下界，M/S_p/kappa提供其余变量的正二次下界。因此K闭且有界，故紧；并且K不含D边界，因为边界H>=2rho>rho。于是K恰等于{q属于D，H<=rho}，尽管D本身是开集，K仍是紧集。

初态必须检查q0属于D和H0<=rho。全局U非凸，不能删掉D检查，也不能认为任意远处分支的低能量点属于K。

从K出发，局部唯一解存在。在首次可能离开D之前H不增；若到达D边界则H至少2rho，与H<=rho矛盾。它也不能从H=rho一侧向外穿出，因为H_dot<=0。因此轨迹保持在紧集K。向量场在K的邻域局部Lipschitz且有界，解可持续延拓，得t>=0前向完备。这个结论关于连续时间ROM解，不能直接推给固定步RK4的每个离散节点。

## LaSalle最大不变集

在K内，H_dot=0要求r=0和p=0，因为Q>=lambda I、lambda/kappa>0、S_p R_p S_p严格正；omega仅由D_omega半正定不能直接推出0。

但若轨迹始终留在H_dot=0集合，r恒为0，故r_dot=kappa omega=0，由kappa>0得到**全部十维omega=0**。进一步omega_dot=0要求grad U(q)=0。D内U在零和子空间严格凸，且0是驻点，故唯一可能为q=0。最大不变集因此是商空间的单点原点。

LaSalle给出K内所有轨迹趋于(q,omega,p,r)=0。正定H及任意足够小子水平集还给出该商平衡点的Lyapunov稳定性，结合吸引性得到局部渐近稳定，K为一个保守吸引域内估计。H_dot是半严格耗散，并非每个非零状态都严格负；不能据上述证明直接宣称指数速率。

恢复40D原坐标时，结论是相对角度趋于0、omega/p/r趋于0；共同角度被商掉。此证明不额外证明共同角度极限存在或其具体值，不把商原点写成原空间唯一平衡点。

## 哪些条件不能放松而仍照抄结论

若lambda=0或kappa=0，上述零耗散集论证失效，kappa=0还使原H_r无定义。若R_p仅半正定，p不一定由H_dot=0被迫为0，须另做可检测性/最大不变集分析；本次不延伸。若势能正边图不连通，消去一个共同角度仍留下无势能方向，mu=0，本rho不能给39D正定内域。若某正边|phi_e|>=pi/2，本条件族不成立，但不等于模型必然不稳定。非严格相位边界可能通过别的加权Hessian条件处理，不属于这次有限命题。

该域本来服务稳定平衡附近，不能拿失稳样本不在域内作为模型失败，也不能用小域覆盖不足否定全部稳定性。反之形式耗散和这个条件命题不能替代原科学L2所要求的具体端点核验、范围和其他门槛。

## 未来一次最小条件核查（未授权、未执行）

若root以后选定一个科学上值得保留的固定端点，可一次性做：

1. 固定源码/导出SHA，核与本文同一field、H、坐标及gamma0；核M、S_p、kappa、lambda、rpp、damping和图权非负/严格正条件；B还核其固定图定义。
2. 只对实际正势能边检查连通性和相位裕量，按固定公式构造w和L_w；得到有可靠数值下界的mu及保守rho。不扫gamma、不优化rho、不调模型使其过证。普通浮点正特征值不能在临界情况下冒称严格下界；界无法分辨则记未决。
3. 如另获准评价某个ROM初态，分别核D与H<=rho并报告数值裕量。无初态误差界时，结论仅覆盖这个已确定的ROM初态；若要覆盖估计误差集合，必须另有有依据的集合界，不能用coarse/fine差当真系统误差界。

以上不是当前新增作业或覆盖率计划。现阶段无实际phi/k审查、无mu/rho数值、无任何初态通过断言。100ms估计r0只构造ROM初态，此内域不转移为真实39母线或高阶FOM的安全保证，不增加原val/test或新FOM权限。root保留后续独立能量挑战及采用权，当前冻结三组继续原协议。
