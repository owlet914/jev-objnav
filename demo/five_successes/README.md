# Five successful Jev trajectories

本目录包含 5 条由真实 Jev API 参与高层决策并最终成功的导航录像。视频是评测器直接导出的三联画：左侧为 RGB 与检测结果，中间为深度，右侧为在线地图和轨迹。

| 文件 | 目标 | 步数 | SPL | Jev 决策数 |
|---|---|---:|---:|---:|
| `01_episode_02_kitchen_shelf.mp4` | kitchen shelf | 196 | 0.212 | 151 |
| `02_episode_11_table.mp4` | table | 40 | 0.687 | 10 |
| `03_episode_14_chair.mp4` | chair | 179 | 0.345 | 145 |
| `04_episode_17_clothes.mp4` | clothes | 65 | 0.687 | 36 |
| `05_episode_18_table.mp4` | table | 88 | 0.775 | 59 |

`clip_01_kitchen_shelf.gif` 至 `clip_05_table.gif` 是供 GitHub README 同时播放的五条独立切片。每条约 11 秒，以介于 40 秒高速剪辑和原速长视频之间的速度播放，并保留第一视角、在线地图、Jev 决策与进度信息。

`jev_objnav_five_successes_showcase.mp4` 是 1 分 41 秒的完整合集，包含片头和每段轨迹的目标、步数、SPL 字幕；5 个带编号的 MP4 是未经剪切的单条轨迹。`showcase_preview.jpg` 是合集封面，`success_trajectory_summary.csv` 提供结构化指标，`success_contact_sheet.jpg` 展示五条轨迹接近终点时的 RGB 观测。

这些是持续评测直到收集到 5 个成功为止得到的成功样本，只证明真实 Jev 闭环可以成功运行，不能单独作为固定测试集成功率。
