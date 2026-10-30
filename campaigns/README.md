# Campaigns（科学假设档案）

一个 Campaign 是一个可被证据支持/削弱/否证的科学假设，可跨多 Run、多会话。
本目录只保存 Campaign 卡与 Run 树，不保存训练日志、进程状态或大体积证据
（那些在 reports/、results/ 与本地存储，按 research_policy 归档）。

    campaigns/<campaign_id>/campaign.md

- Campaign 卡模板见 templates/campaign.md；Run 协议/闭合见
  templates/run_protocol.md、templates/run_closure.md；机制卡见
  templates/mechanism_card.md。
- 每个 Run 结束在 Campaign 卡的 Run 树追加一行，并分别记录代码/模型/科学三个决定。
- 一次 Run 没超过可用模型，不等于 Campaign 失败；只要减少重要不确定性就沿同一
  假设推进。换编号不重置同一路线的延长次数，也不能清除负面证据。
- 历史实验保持 reports/ 原路径与身份，不迁移、不重写；Campaign 卡只链接其证据。
