# Jev Obj 轨迹控制演示

本目录包含轨迹生成与 MPC 控制示例；先按 [主文档](../README.md) 准备 ROS、Habitat 和视觉模型。在 `jev_obj/` 工作空间下运行：

```bash
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
roslaunch exploration_manager rviz_traj.launch
roslaunch exploration_manager exploration_traj.launch
```

视觉服务各在单独终端运行：`python -m vlm.detector.grounding_dino --port 12181`、`python -m vlm.itm.blip2itm --port 12182`、`python -m vlm.segmentor.sam --port 12183`、`python -m vlm.detector.yolov7 --port 12184`。仿真侧可使用 `python habitat_vel_control.py` 与 `python real_world_test_example/real_world_test_habitat.py`。每个命令需在匹配的环境中运行。