# Jev Obj

Jev Obj 是一个把 **Jev 决策**接入目标导航的实验项目。ROS 规划器负责感知、建图、生成可达目标和执行路径；Jev 根据规划器提供的实时信息，从已有候选目标中选择下一步。Jev 不生成坐标、轨迹或控制指令。

## 工作流程

```text
Habitat / 相机 RGB-D
  → 视觉检测与语义融合
  → 地图、物体、探索边界
  → 规划器计算可达路径和候选目标
  → 结构化状态发送给本地 Jev 服务
  → Jev 返回候选 ID / hold
  → 校验后由 ROS 规划器执行，继续观察并循环
```

### 1. 提取导航信息

`jev_obj/` 中的规划器使用当前传感器观测和内部地图，生成一份 JSON 快照，主要包括：

- 机器人位置、地图坐标系、版本，以及机器人周围的局部占用和语义栅格；
- 已观测物体的位置、目标类别置信度与观测次数；
- 探索边界、语义匹配、信息增益；
- 规划器已计算的候选目标 ID、可达路径、距离和语义分数。

任务目标名称、相关物体和房间先验也可以作为上下文。**房间先验不等于已观测到的房间**；当前没有房间分割结果时，`rooms` 为空。在线决策只使用运行时信息，Habitat 的目标视点和最短路径真值仅用于评估，不传给 Jev。

### 2. 接入 Jev 决策

C++ 规划器通过 HTTP 将快照交给 `nav_jev_bridge/`。服务校验数据，只保留可达且无碰撞的候选，再通过 OpenRouter Decisions API 调用 Jev。可选 `candidates`、`spatial`、`full` 三种输入档位，用于比较给模型的信息量。

Jev 的回答只能是某个候选 ID 或 `hold`。服务会检查选择、置信度和概率；C++ 再核对 episode、地图版本、坐标系和候选 ID。若请求超时、结果无效或 Jev 选择 `hold`，规划器回退到原有策略。路径搜索和动作执行始终由 ROS 侧负责。

完整的原生 Ubuntu、WSL2、无桌面服务器与轻量环境配置，见 [环境配置指南](docs/ENVIRONMENT.md)。

## 目录与快速检查

- `jev_obj/`：ROS catkin 工作空间、Habitat 评估与感知代码。
- `nav_jev_bridge/`：Jev 请求、输入校验和本地 HTTP 服务。
- `tools/`：OVON 数据准备、轨迹回放和结果汇总；`tests/`：回归测试。

在项目根目录的 Python 3.10+ 环境运行以下命令，可验证接口和示例输入：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m nav_jev_bridge.cli examples/snapshot.json --provider demo
```

真实调用需在服务进程环境中设置 `OPENROUTER_API_KEY`，然后从项目根启动：

```bash
python -m nav_jev_bridge.server --provider openrouter --state-profile full
```

规划器还需要 ROS Noetic、Habitat、HM3D/OVON 数据、视觉服务与模型权重。进入 `jev_obj/`，编译并启用决策分支：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
roslaunch exploration_manager exploration.launch jev_enabled:=true habitat_config:=$(pwd)/config/habitat_eval_ovon.yaml
```

另一个终端在相同工作空间运行 `python habitat_evaluation.py --dataset ovon`。数据、权重、密钥、日志与编译产物不随仓库上传。`demo` 仅用于接口检查，不能代表 Jev 的实际导航效果。

**当前进展：** Python 测试与 ROS 编译已通过；已有 1 次完整的 HM3D-OVON pilot episode，成功 0/1。该次使用临时视觉回退服务，不能用来评价完整感知系统或最终导航成功率。

## 参考

本项目的 ROS 导航、建图与探索框架参考并改造自 [ApexNav（Robotics-STAR-Lab）](https://github.com/Robotics-STAR-Lab/ApexNav)。我们在此基础上加入运行时信息提取、Jev 候选决策、结果校验和回退机制；原源码许可证见 `jev_obj/LICENSE`。