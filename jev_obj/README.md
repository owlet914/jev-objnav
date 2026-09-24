# Jev Obj 导航工作空间

本目录包含 ROS Noetic catkin 源码、Habitat 评估、视觉与目标先验模块。项目入口、Python 决策服务、测试命令和目前的验证状态见仓库根目录 `README.md`。

在本目录中准备 `data/`、外部视觉模型依赖和权重，运行 `catkin_make` 后 `source devel/setup.bash`。仿真配置为 `config/habitat_eval_ovon.yaml`，规划器入口为 `roslaunch exploration_manager exploration.launch jev_enabled:=true`。ROS 参数接口使用 `/jev_obj/jev/*`，Jev 服务默认运行在本机 `127.0.0.1:8765`。

`real_world_test_example/` 是轨迹控制演示，未用于当前 OVON pilot 成功率验证。