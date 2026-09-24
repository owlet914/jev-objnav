# Jev Obj 环境配置与运行

本文从**接口检查 → ROS 编译 → Habitat 单步 → Jev + 导航**逐级验证。以下命令默认在仓库根目录执行；标注 `cd jev_obj` 的命令需切换到该目录。先前版本在 Ubuntu 20.04 / ROS Noetic 上完成过单次 pilot；本次更名后的 `jev_obj/` 已重新编译并通过接口测试，**尚未重新运行完整 episode**。其它平台是可选配置路径，必须逐级验证，不能把示例检查当成完整导航成功。

## 1. 按机器选择路径

| 机器与目的 | 建议配置 | 可验证内容 |
| --- | --- | --- |
| Windows / macOS / Linux、没有 NVIDIA GPU | Python 3.10+；只安装根目录的桥接包 | 单元测试、JSON 决策协议、示例 HTTP 服务；**不能直接跑完整 ROS/Habitat 导航** |
| Ubuntu 20.04 x86_64、带 NVIDIA GPU 的工作站 | ROS Noetic、Habitat-Sim/Lab 0.3.1、视觉模型 | 完整导航的首选路径；需自行准备数据与权重 |
| Windows 11 + WSL2 + NVIDIA GPU | WSL2 内使用 Ubuntu 20.04；或保留 Windows/WSL 22.04 作为 Jev 开发环境，另外用 Ubuntu 20.04 跑 ROS | 可按 Ubuntu 路径逐级验证；视觉/GPU 需额外检查 |
| 无桌面的 Linux GPU 服务器 | Ubuntu 20.04、ROS Noetic、Habitat headless/EGL | 无 RViz 的仿真与批量评估 |
| 新款 NVIDIA GPU（包括 Blackwell）或显存紧张的单卡机器 | ROS/Habitat Python 3.9 和视觉服务 Python 3.10+ 分开；视觉服务使用适配显卡的 PyTorch wheel | 先测 GPU 推理，再测整套服务；多模型同时驻留显存未保证 |

ROS Noetic 已于 **2025-05-31** 结束官方支持。Ubuntu 22.04/24.04 不要把 Noetic 的 Ubuntu 20.04 二进制包当作原生安装；可用独立的 20.04 虚拟机、WSL 发行版或容器。目前工程是 ROS 1 catkin 工作空间，尚未移植到 ROS 2。

## 2. 所有机器先做接口检查

从根目录创建单独的 Python 3.10+ 环境（下面示例使用 Conda；已有 venv 也可）：

```bash
conda create -n jev-obj-bridge python=3.10 -y
conda activate jev-obj-bridge
python -m pip install -e .
python -m unittest discover -s tests -v
python -m nav_jev_bridge.cli examples/snapshot.json --provider demo
python -m nav_jev_bridge.server --provider demo
```

在另一终端执行 `curl http://127.0.0.1:8765/health`。`demo` 是离线启发式检查，不会调用 Jev，也不衡量导航效果。Windows 可在 PowerShell 的 Python 环境执行相同的 `python -m ...` 命令，HTTP 健康检查可用 `Invoke-RestMethod http://127.0.0.1:8765/health`。macOS、ARM 或 CPU-only 设备建议先止步于此层；完整 ROS 1/Habitat 环境需要额外移植与验证。

## 3. Ubuntu 20.04：ROS 与规划器

在 Ubuntu 20.04 x86_64 上按 ROS Noetic 的官方安装流程配置软件源，再安装本项目需要的包。以下是**已配置 ROS 软件源后**的依赖示例：

```bash
sudo apt update
sudo apt install build-essential cmake git python3-catkin-pkg python3-empy \
  ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-pcl-ros ros-noetic-rviz ros-noetic-cmake-modules \
  libeigen3-dev libpcl-dev libopencv-dev libompl-dev libarmadillo-dev \
  libcurl4-openssl-dev
source /opt/ros/noetic/setup.bash
rosversion -d
```

编译还需要 OSQP 0.6.3 与 OsqpEigen 0.8.1。按上游源码分别编译安装，确认 CMake 能找到 `OsqpEigen`：

```bash
mkdir -p ~/jev-obj-deps
cd ~/jev-obj-deps
git clone --recursive --branch v0.6.3 https://github.com/osqp/osqp.git
cmake -S osqp -B osqp/build -DBUILD_SHARED_LIBS=ON
cmake --build osqp/build -j 4
sudo cmake --install osqp/build
git clone --branch v0.8.1 https://github.com/robotology/osqp-eigen.git
cmake -S osqp-eigen -B osqp-eigen/build
cmake --build osqp-eigen/build -j 4
sudo cmake --install osqp-eigen/build
sudo ldconfig
```

回到仓库，编译 catkin 工作空间：

```bash
cd /path/to/jev-obj/jev_obj
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release -j4
source devel/setup.bash
rospack find exploration_manager
```

`/path/to/jev-obj` 要替换成你自己的仓库路径。工作空间的 `build/`、`devel/` 均为本地产物。若只想验证 C++，到这一步即可，不需要安装大数据包。

## 4. Habitat 与 Python 环境

完整仿真建议使用 **Python 3.9、Habitat-Sim 0.3.1、Habitat-Lab 0.3.1**；不要用最新 Habitat 主分支替代 0.3.1。桌面机器可以从下列最小 Conda 环境开始；无桌面的服务器将 `withbullet` 后面加上 `headless`：

```bash
conda create -n jev-obj-sim -c aihabitat -c conda-forge --override-channels \
  python=3.9 habitat-sim=0.3.1 withbullet numpy=1.23.5 -y
conda activate jev-obj-sim
python -c 'import habitat_sim; print("habitat_sim ready")'
```

仓库中的 `jev_obj/jev_obj_environment.yaml` 是一台 Linux 机器的详细环境快照，包含特定构建号；跨系统不保证原样求解成功，优先按上面的命令建立环境，然后逐项补依赖。Habitat-Lab 以**固定 v0.3.1** 源码安装：

```bash
cd /path/to/jev-obj/jev_obj
mkdir -p .deps
git clone https://github.com/facebookresearch/habitat-lab.git .deps/habitat-lab
git -C .deps/habitat-lab checkout tags/v0.3.1
python -m pip install -e .deps/habitat-lab/habitat-lab
python -m pip install -e .deps/habitat-lab/habitat-baselines
python -m pip install salesforce-lavis==1.0.2
python -m pip install -e .
```

安装模型或项目包后重新检查 `numpy`、`torch` 和 `habitat_sim` 的 import；Python 依赖可能因当前软件包仓库或显卡而冲突，应以这些验证结果为准。ROS Noetic 的系统 Python 是 3.8，Habitat 这一层使用 Python 3.9：运行评估时先 source ROS 与 `devel/setup.bash`，再激活 Conda；若系统 ROS 路径让 Conda 的 NumPy 被覆盖，在 `jev_obj/` 目录运行 `python ../tools/run_focal_python.py habitat_evaluation.py --dataset ovon`，让 Conda 包优先于 ROS 的系统 Python 路径。

## 5. 数据与模型：仅在完整导航时准备

数据不随 Git 仓库分发。自行获取有访问权限的 HM3D 场景包与 OVON episode 包，放到仓库根目录（下列命令假设文件分别名为 `hm3d.zip`、`hm3d_ovon.zip`），从仓库根目录执行：

```bash
python -m tools.extract_ovon_val
python -m tools.prepare_ovon_habitat031
python -m tools.prepare_ovon_pilot
```

它们会在 `jev_obj/data/datasets/objectnav/ovon_small/` 生成 pilot/dev，并把所需 HM3D 场景提取到 `jev_obj/data/scene_datasets/`。配置 `jev_obj/config/habitat_eval_ovon.yaml` 默认使用 pilot。**只下载和使用获得授权的数据**；验证集真值仅供 Habitat 评分，不应进入 Jev 在线输入。数据就绪后，在 `jev_obj/` 下可试 `python ../tools/smoke_ovon_episode.py`；该命令只加载场景并走一步，不会启动导航策略。

完整感知还要把 `groundingdino_swint_ogc.pth`、`mobile_sam.pt`、`yolov7-e6e.pt` 放到 `jev_obj/data/`。下载来源和许可按各模型仓库要求操作；BLIP2 ITM 可能在首次运行时单独下载缓存。代码要求 GroundingDINO 源码位于 `jev_obj/.deps/GroundingDINO/`，YOLOv7 源码位于 `jev_obj/.deps/yolov7/`。在 `jev_obj/` 下可先取源码，再按各模型项目说明安装其 Python/CUDA 依赖：

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git .deps/GroundingDINO
git clone https://github.com/WongKinYiu/yolov7.git .deps/yolov7
```

四个服务从 `jev_obj/` 分别启动：

```bash
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```

每个命令各占一个终端；先逐个验证实际加载和推理，再考虑同时运行。单卡显存不足时可能需要减少并行模型或将某些服务放在 CPU；当前代码默认通过 `localhost` 联系视觉服务，跨机器部署还需要修改服务地址配置，不能仅改启动命令。

## 6. 新款 NVIDIA GPU、WSL2、无桌面服务器

- **较新 NVIDIA GPU：** 先查看驱动、PyTorch CUDA 版本与计算能力：`nvidia-smi` 和 `python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability() if torch.cuda.is_available() else "CPU")'`。例如 PyTorch 官方提供 **2.7.1 + CUDA 12.8** 的 Python 3.10+ wheel，可用于单独的视觉服务环境；它**不能直接安装进 Python 3.9 的 Habitat 0.3.1 环境**。示例命令（仅为 PyTorch 环境，还须按视觉模型项目安装其余依赖）：

  ```bash
  conda create -n jev-obj-vlm python=3.10 -y
  conda activate jev-obj-vlm
  python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  python -c 'import torch; x = torch.ones(4, device="cuda"); print(x.sum().item(), torch.cuda.get_device_name())'
  ```

  之后逐项加载视觉模型，核对其依赖是否与这一版 PyTorch 兼容。显卡兼容性需以实际驱动和 wheel 的支持矩阵为准。
- **Windows + WSL2：** 在 PowerShell 查看可安装发行版：`wsl --list --online`；若列表提供 Ubuntu 20.04，可执行 `wsl --install -d Ubuntu-20.04`。若已不提供该发行版，可用独立 Ubuntu 20.04 虚拟机/容器，或按 Microsoft 文档导入自行取得的可信 Ubuntu 20.04 根文件系统。安装兼容的 Windows NVIDIA 驱动后，在 WSL 内确认 `nvidia-smi`，不要在 WSL 内再装 NVIDIA Linux 显示驱动。推荐将仓库存放在 WSL 的 Linux 文件系统以减少大数据读写开销；随后按 Ubuntu 20.04 路径构建。若只用 Windows 运行 Jev 服务而 ROS 在 WSL，默认 `127.0.0.1:8765` 的跨环境可达性须单独确认；最简单是两者先在同一 WSL 环境内运行。
- **无桌面服务器：** 安装 Habitat-Sim 的 `headless` 变体，设置有效 EGL GPU 环境后执行单步 smoke；不需要启动 RViz。CPU-only 机器可以先完成第 2、3 节与有限的场景加载检查，不能预期四个视觉模型及完整导航具有实用速度。

## 7. 顺序启动并确认结果

1. 从仓库根启动 Jev 服务：先在当前 shell 私密设置 `OPENROUTER_API_KEY`，再运行 `python -m nav_jev_bridge.server --provider openrouter --state-profile full`。服务默认监听 `127.0.0.1:8765`，`/health` 应返回 `ok`。不要把 key 写入仓库或日志。
2. 在 `jev_obj/`、ROS 环境中运行 `roslaunch exploration_manager exploration.launch jev_enabled:=true habitat_config:=$(pwd)/config/habitat_eval_ovon.yaml`。
3. 在另一个终端、`jev_obj/` 的 ROS + Habitat 环境中运行 `python ../tools/run_focal_python.py habitat_evaluation.py --dataset ovon`；视觉服务需事先全部就绪。
4. 检查 ROS 日志中的 Jev 选择/回退以及 `jev_obj/videos/` 下的 episode 结果，如将各组结果写入 `jev_obj/videos/ovon_trials/<组名>/`，再用 `python -m tools.summarize_ovon_trials` 从仓库根汇总已经完成的实验；默认 `videos/test_ovon_{split}` 不在这个汇总目录内。接口检查、Habitat 单步和真实 Jev 请求都不等于 episode 成功。

## 官方资料

- [ROS Noetic 生命周期](https://www.ros.org/blog/noetic-eol/)；[ROS Noetic 安装说明](https://wiki.ros.org/noetic/Installation/Ubuntu)。
- [Habitat-Sim 安装说明](https://github.com/facebookresearch/habitat-sim/blob/main/README.md)；[Habitat-Lab 仓库](https://github.com/facebookresearch/habitat-lab)。
- [PyTorch 各版本安装命令](https://pytorch.org/get-started/previous-versions/)；[NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/)；[Microsoft WSL 安装与文件系统建议](https://learn.microsoft.com/en-us/windows/wsl/filesystems)。