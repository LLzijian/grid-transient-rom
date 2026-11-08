# Agent 任务：<task_id>

- role / owner / 实际模型回执：
- question：此任务帮助哪个具体科学或执行决定？
- inputs：文件/证据与 commit/hash；审查对象必须冻结。
- outputs：最小交接产物及可审查依据。
- write_paths：精确相对路径；只读用 []；不授予整库通配写入。
- depends_on：任务 ID；共享接口与单一 writer。
- resources：GPU/CPU/RAM 等实际占用；agent slot 由宿主管理。
- status：pending / running / completed / failed / blocked / cancelled / paused。
- acceptance：收到什么证据才完成；未核对的能力不宣称。
- stop_condition：何种阻碍/范围变化需要上交；不能越过全局暂停。

交接只含结论、证据、未决点和下一动作。不得以“已委派”冒充工作已完成。
