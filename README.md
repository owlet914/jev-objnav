# JEV-ObjNav

JEV-ObjNav 将 **Jev 决策模型接入开放词汇目标导航**：机器人从 RGB-D 观测中提取目标、空间和地图信息，在本地完成建图、候选生成与可达性验证，再把压缩后的区域图和候选动作交给 Jev。Jev 综合环境语义、探索进展和目标证据，在“继续探索”和“接近目标”之间选择下一步；本地规划器把这个选择转换为可执行路径，机器人执行后更新观测并进入下一轮决策。这样，Jev 负责利用多源视觉信息进行高层判断，导航系统负责几何规划与动作控制，形成从感知到执行的完整闭环。

## 🎬 Jev 导航 Demo

我们测试了 18 个场景，成功完成其中 5 个（**5/18**）。下面是这些成功场景的 Demo 示例，包含第一视角、在线地图、Jev 决策和导航进度；点击片段可查看完整轨迹。

<table>
  <tr>
    <td width="50%" align="center">
      <a href="demo/five_successes/01_episode_02_kitchen_shelf.mp4"><img src="demo/five_successes/clip_01_kitchen_shelf.gif" alt="Kitchen shelf navigation trajectory"></a><br>
      <b>01 · Kitchen shelf</b><br>
      196 steps · SPL 0.212 · 151 Jev decisions
    </td>
    <td width="50%" align="center">
      <a href="demo/five_successes/02_episode_11_table.mp4"><img src="demo/five_successes/clip_02_table.gif" alt="Table navigation trajectory"></a><br>
      <b>02 · Table</b><br>
      40 steps · SPL 0.687 · 10 Jev decisions
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <a href="demo/five_successes/03_episode_14_chair.mp4"><img src="demo/five_successes/clip_03_chair.gif" alt="Chair navigation trajectory"></a><br>
      <b>03 · Chair</b><br>
      179 steps · SPL 0.345 · 145 Jev decisions
    </td>
    <td width="50%" align="center">
      <a href="demo/five_successes/04_episode_17_clothes.mp4"><img src="demo/five_successes/clip_04_clothes.gif" alt="Clothes navigation trajectory"></a><br>
      <b>04 · Clothes</b><br>
      65 steps · SPL 0.687 · 36 Jev decisions
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <a href="demo/five_successes/05_episode_18_table.mp4"><img width="50%" src="demo/five_successes/clip_05_table.gif" alt="Second table navigation trajectory"></a><br>
      <b>05 · Table</b><br>
      88 steps · SPL 0.775 · 59 Jev decisions
    </td>
  </tr>
</table>

五个成功样本的平均 SPL 为 **0.541**。完整视频和结构化指标见 [`demo/five_successes/`](demo/five_successes/)。

## 🔄 方法概览

```text
RGB-D 观测
  ├─ 目标检测、分割、图文匹配
  ├─ 占用地图 + 语义价值地图 + 对象地图
  ├─ 前沿、对象和可达路径
  └─ 区域图表示
       ├─ 区域、对象、前沿与稀疏可达关系
       ├─ 机器人状态、历史进展与视觉证据
       └─ 候选动作与路径信息
                         ↓
                   Jev 选择候选 ID
                         ↓
            本地校验 → 执行 → 新观测 → 再决策
```

Jev 负责根据语义信息和全局进展选择下一步，本地规划器负责几何计算、路径搜索和动作执行。

## 👁️ 从观测中提取的信息

每个导航周期取得 RGB、深度、相机位姿和目标名称。视觉服务提供开放词汇检测框、分割掩码、标签以及图文匹配分数；深度与位姿把二维检测投影到空间中，并在多帧之间融合。系统维护三张互补地图：

1. **占用地图**：记录自由空间、障碍、未知区域、膨胀障碍与安全间距，供碰撞检查和路径搜索使用。
2. **语义价值地图**：把画面与目标、房间先验和相关物体的语义相关性累积到空间中，为探索区域和前沿估计价值。
3. **对象地图**：把跨帧检测融合成对象簇，维护位置、边界、类别证据、置信度、观测次数、新鲜度和多视角一致性。

地图层生成两类候选：`explore_frontier` 前往已知与未知空间的边界获取新信息；`approach_object` 接近已经观察到、且本地验证可达的目标对象。候选同时携带本地 A* 路径状态、距离、语义证据和唯一 ID，Jev 返回候选 ID 后由本地规划器完成坐标映射和路径执行。

## 🗺️ 把三张地图转换成区域图

三张地图本身由高分辨率栅格、点云和跨帧检测记录组成。我们把这些信息转换成语言模型更容易阅读的区域图：每个区域作为一个节点，节点记录空间边界、语义价值、对象和前沿，节点之间记录已经验证的可达关系；像素级地图和精细几何继续留在本地规划器中。这样既保留了高层决策需要的空间结构，也避免把大量重复格子和点云直接写入请求。

区域化的另一个目标是适配 Jev 的 **32K 上下文窗口**，同时减少输入 token、API 费用和通信延迟。Jev 接收与决策直接相关的区域、对象、前沿和路线信息，本地规划器继续维护原始传感器数据与精细几何。

系统从占用地图中取已知、自由且可通行的格子，再按固定空间块和局部连通性形成区域节点；被墙隔开的空间不会因为处于同一方块而合并。每个区域维护：

- 稳定 ID、二维边界框、可通行锚点和自由面积；
- 是否访问、访问次数和最近访问步数；
- 语义价值的 `mean / p90 / max / valid_area_fraction`；
- 区域内对象的中心、边界框、类别证据、置信度、观测次数和新鲜度；
- 区域内前沿的位置、语义分数、信息增益和可达状态。

区域之间记录当前地图验证过的稀疏可达边。机器人状态包含当前区域与 `{x, y, yaw}`；候选包含目标区域、局部路径长度和 `route_region_ids`，完整路径点由本地规划器维护。

### 如何压缩 Token

区域化把本地地图中的重复几何转换成稳定、可比较的导航信息，同时保留本地规划器使用的全精度地图。

- 用区域边界框和统计量替代逐格地图；
- 用对象簇信息替代检测像素和三维点；
- 用前沿中心、边界长度、价值和可达性替代前沿格子集合；
- 用稀疏区域边和区域路线替代完整路径坐标；
- 坐标和分数统一保留合理精度，本地执行继续使用全精度数据；
- 保留规划器验证后的候选，以及与决策直接相关的置信度、新鲜度、历史进展和来源阶段。

区域图让 Jev 能直接理解：机器人在哪里、哪些区域已经探索、哪些区域值得继续查看、观察到了哪些对象，以及当前有哪些动作可以执行。当前协议为 `jev-region-graph/1.1`，示例见 `examples/region_graph/`。

## 🧠 Jev 如何参与决策

桥接服务把区域图、机器人状态和候选列表组织成 OpenRouter Decisions 请求。候选分为两类：`explore_frontier` 用于前往未知空间边界，`approach_object` 用于接近已经观察到的目标对象。Jev 综合区域价值、对象置信度、观测新鲜度、多视角一致性、历史进展和路径代价，返回下一步候选 ID；本地规划器根据对应候选完成路径执行。

接近对象后，系统会用新的视觉观测检查目标位置和多视角证据，再确认是否完成任务；如果当前观察还不足以确认目标，导航会继续寻找更可靠的观察位置。

## 项目结构

- `jev_obj/`：ROS/catkin 规划器、Habitat 评估、地图和视觉感知；
- `nav_jev_bridge/`：区域图校验、Jev 请求与本地 HTTP 服务；
- `tools/`：环境检查、启动、评测、审计和结果汇总；
- `tests/`：协议、区域图、感知元数据和运行审计测试；
- `examples/region_graph/`：区域图请求与响应示例；
- `docs/`：环境、配置和协议说明。

## 快速开始

```bash
conda env create -f environment.yml
conda activate jev-obj-bridge
python -m pip install -e .
python -m unittest discover -s tests -v
python -m nav_jev_bridge.cli examples/snapshot.json --provider demo
```

完整导航需要 Ubuntu 20.04、ROS Noetic、Habitat 0.3.1、HM3D/OVON 数据和四个真实视觉服务。请按 [环境与配置指南](docs/SETUP.md) 分层安装，并复制 [.env.example](.env.example) 配置本机路径。

## 参考

本项目在导航建图、探索和候选规划方面参考了 [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) 的方法思想，并在此基础上设计了面向 Jev 的区域图信息提取、上下文压缩、高层决策与到达复核流程。
