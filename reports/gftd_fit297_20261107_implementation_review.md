# Run3 整体实现独立审查

结论：冻结源码中的必要实现问题已修复，主体与最终设计一致；当前状态 `implementation_accepted_pending_joint_receipt`。一次11文件联合工程回执须在root释放前补入。此报告不启动fit297读取、拟合、实际证书/Kron/谱或新FOM，也不预设模型晋升。

## 冻结绑定

协议 `CA22743C53FE4FE398960E111838A373F8C4686EAB0974C5DAF6D9C28435062F`；设计审查 `E9CF79E5F6A82101E811AC722CFB4192066378C6FA90864C9FE9CBB4DD864B6A`；父Run2归档 `66f1308c25ee38c3d7217bd86ee43b61a042a5b6`。

| 实现 | SHA256 |
|---|---|
| `src/kiki/models/base.py` | `95F5D5A002CEAE3A48CE1D7F82000355DE87361C09AA56C48EAB7E44D5093638` |
| `src/kiki/models/gftd_kuramoto.py` | `8A981125BBA23E1B771E29B17A22B2E585F82A05FB26A4D64A87F2F86E029F35` |
| `src/kiki/models/gftd_discovery.py` | `9245B0726CB3FCA15D3248106B43BD1673ED6607ED3A852B67C533F768A1B27F` |
| `src/kiki/models/gftd_fitting.py` | `57EFFAAB858200BF77A33DE6661210104BB01BAE2DEF4E8DB718C08254D982F4` |
| `src/kiki/models/gftd_fit297.py` | `43F937B6973490C771AEA2E1AC8694A2C356A48FB2F26337A73C9E991C6E334B` |
| `src/kiki/physics/kuramoto_functional.py` | `5A709BC2B5B984FBDD4C376F69852E0A53A12929DEE3F525A453B5934CAEC021` |
| `experiments/gftd_fit297_20261107.py` | `7E680ABF096D2296903BBB4ACE52A6018C27C3F7551874A22E1249B1D13DFFC4` |
| `experiments/gftd_fit297_l1_20261107.py` | `674708294A71338487856063799D189D0F693A3055C4768D7D70890CA9B79129` |
| `experiments/gftd_fit297_analysis_20261107.py` | `2FAA8CDD83A98F71C044E20983B2B08D0F394F76B64C15B24F2C63617DAB929E` |

对应新测试依次绑定：runner `4929F1EEE08C70434D09D51BE7BC4364CEE4A5D8FD0FACA79CFED70E76136724`、L1 `A3E694660936A7E18C57615A26128E25385B9C84F1EEA1E008C9AD7F9F03A933`、engine `8D331D31C5AED79D44150BC01839CD48CC8C776E4264E9D7ACC79B9C6CB1E9A6`、analysis `55F1E706C4F25BC04DFBF1526B3A6200B97D5FAE1D4DCA9EB0F6062A3ED0256F`。

已接受组件详见analysis_review（`99C312D974B982EF43CC09BD34B200F03C1D5DB5DB8CF67C22373BA9558CE90D`）和engine_review（`6A35B948E743A0A93093266D6703803B471BE8DE45414B5D8575F4A2B261010E`）。本次读取源码/测试并作最小diff核验；独立运行仅已记录的15个标量Taylor代数核查，没有真实fit、事件读取、RHS积分或实际模型分析。工程mock不是独立科学样本。

## 四项必要修复均已闭合

1. **细格信息分析异常。** `_audit`补ValueError/RuntimeError，主格成功后细格失败返回unknown，定向mock保留47尝试/46完成及局部结果，不重试。engine v1原字节和原95项资格保留，后续17项定向资格是新增回执，不冒称重跑全部95。
2. **冻结常数直接JSON提取。** runner改从hash绑定R_N有效buffers及reference×exp(raw)提取；不加载父对象、不调用其matrix_exp。v1确实曾加载父对象的工程事实保存在 `pre_review_api_v1/`，不回写为“从未发生”；未因此产生新的科学轨迹。
3. **ROM RHS预算。** 197上限错误会拒绝合法分析；现上限217，包含最多20轮当前点检查＋最后1次检查＋两格196次差分RHS。198/217允许、218拒绝的mock通过。没有增加Newton轮数、候选或科学机会。
4. **先登记成本再I/O。** fit正常/typed失败、初始化正常/typed失败的实际计数先合入receipt，再写模型、npz或诊断；未来完成数和model-only分析计数也在产物I/O前保存到内存。四类首次I/O失败mock验证最终回执保留已用成本且不重试。不能据此承诺I/O根因消失，任何新中断仍按已有记录对账，不自动重放。

API/runner最终54项工程资格回执 SHA `47FDAD7469FF48BC1C4EFB3F72EF40DD4B1DBF27A752DA41A8C7A9DA6880DD28`（167.77秒）；仅按冻结回执核对，本次未重跑。

## 完整调用合同

- **因果/数据：** root release先绑定父归档、协议/设计、全部实际导入项目源、环境和必要资产；独占claim在fit缓存payload读取前建立。只走登记train fit297入口；ID/标签和分类元数据留runner，模型严格五键/四键，不传未来Pm、隐态或故障。非法输入保留原ID/原因，不挑替补；Unicode排序及固定最多24索引在拟合前持久冻结。旧test谱系继承明确，本轮不称独立确认。
- **API/存档：** RunContext专用配置显式分派新fit297版本，旧legacy和Run2严格格式仍走原路径；局部副本验证后更新self，失败不改父模型。新config保存尺度、Q、搜索盒、k_ref阈值及来源/物理指纹；parent_initializer不遗留到新因果模型。private_details实际另存并绑定hash，未来初始化只用合法历史及真p，完整50维。
- **拟合/信息：** 全体静态准备一次、失败缓存不被第二模型重算；24范围另用对应总行数和共同记忆截止。候选a随参数重推，精确指数列、层级后向、一次阈值/去偏和重新推断后的history接受一致。单事件不要求45/90列可辨，shared恢复门与部署后信息分开；null/raw/先验保留。SVD/投影额度封顶，分析unknown不修选模型。
- **初始化/冻结：** 每合法包独立simple600→smooth1200，先过stage1自身资格；不复用代理隐态或细格再优化。同点损失守卫、硬p和原尺度数值门保留；先冻结两模型及全部初始化，再覆盖分析、双步自治预测和评分，不因未来表现改模型或起点。
- **L1：** 精确公式及字段绑定L1合同 `2D09F180266875A6100C15C61CC915FC5A2E7752D1DE641DCE6BE603D6C74252`。NumPy转写q/risk、池化共同ω/p、1/2/5/8秒和真实≤15秒索引正确；全部50维/内部细步有限性参加资格。numeric_unknown、延迟未覆盖、失败与有效时窗未告警分别记录；数值合格但未settled且未告警可计漏检，同时保留unsettled类，不称稳定。时机只在真/预测trigger都在同实际范围时比较。267/30/297及594原分母不变，全范围缺项为null，匹配子集仅辅助。
- **L2/L3：** 有理因子定义同一实数ODE，固定舍入中心/13半径/有限包络、无ε时仅ROM点证书；Kron为已声明PQZ端电压参考。FOM只用已有699维descriptor的13模式；ROM只删共同θ、完整ω，同单位/质量权重及inf Hungarian全配对，未知/未匹配不删。均report-only，不变成L1晋升门。

最大登记工作量仍为57337代数项、1572优化任务/1146000实际残差、49229640登记RK4步，完整与代理预算不转移。信息1878 SVD/585列投影/4000记忆分解独立封顶；model-only次数另列。原子小产物、阶段回执、失败成本及无自动resume保持，最终源/释放哈希检查不能省略。

## root释放前的最终绑定

联合工程资格须绑定上述最终源，不能复用v1失败源码的通过回执。release中 `fom_expected_identity` 应由静态audit字段加 `equilibrium_verification_reference=static_identity_audit.json#sha256=EAC00…` 生成，asset清单强制绑定该完整audit；同case/机组/全局平衡的保留证据链可支持此身份，不能只靠未经绑定布尔值。root已明确采用该生成方式。静态prepare/source/schedule链、数据asset哈希和全部动态导入依赖须由最终release覆盖；源环境不一致即不执行。

代码决定：修订实现可保留，待一次联合回执闭合后可交root精确释放；模型决定：不晋升；科学决定：仅登记真实train条件下同期比较，不预判调制恢复、全局优化充分、L2点覆盖或FOM结构对应。本报告不给额外研究资格要求。

## 联合回执与最终接受

唯一11文件联合资格 `integrated_qualification.json` 已完整停写，SHA256 `2AE83F7C67E6D2E3B84B6929FD5BDF600BD7F4A4C54A2516FC08765DC161F100`。独立核原stdout为 **177 passed、0 failed/0 skipped、exit0**，pytest 298.66秒，宿主313.5609633秒；21项源码/测试的执行前、执行后及当前哈希全部相同。stdout/stderr原字节哈希与回执一致，stderr为空；原命令文本与回执命令一致。这是一次联合调用，未发生科学执行、fit297事件读取或真实模型证书/模态计算，不把各阶段资格重复相加。

release草案 SHA `ED4CA5764031B1C31B0095F263F4BE63645F14BE981492BBF7BC661EBD98C749` 的51项目源及20个非fit缓存资产已独立核当前哈希一致；第21个fit_events.npz未打开，其登记哈希须按计划在独占claim后核验。草案released=false；父归档、采用回执、环境和FOM expected_identity绑定一致，后者逐项等于静态audit加完整EAC00…路径/hash引用。最终宿主将显式设置并记录三项线程环境为1；该外层字段不在runner.environment()自动比较中，须由启动回执记录，不改源码。

新增纯静态 `equilibrium_metadata_crosscheck.json` SHA `63298B83E763492DB32BFF90BF25DF88F00BCBE713F18502D58DBDC67D7A7A09`，显示旧quartic preparation的raw equilibrium delta摘要为 `6F12D85C17487664C2DFFCAFB18306FC9E7BC639A1B8616AC2F35937B16EC5B5`，与已核HDF5原平衡一致；结合旧loader逐事件array_equal合同，支持新rawdelta哈希守卫不会把仅共同角规范差的模型δstar混作样本非法。没有为此读fit缓存或事件。

全部必要修复与工程资格闭合，最终状态 **`eligible_for_root_release`**。允许root绑定本最终报告、联合资格及完整源/资产/环境后，释放协议内唯一Run3 action；本审查本身不启动计算。代码可用与后续数值/优化/机制结果分别判断，模型仍不晋升。报告至此停写。
