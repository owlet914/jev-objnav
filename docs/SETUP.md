# JEV-ObjNav 环境与配置指南

完整系统分为 Jev 桥接、ROS/Habitat 导航和视觉模型服务。建议分环境运行，避免 Habitat、ROS 和新显卡 PyTorch 互相锁死版本。

## 部署选择

| 机器 | 推荐配置 | 能运行的部分 |
|---|---|---|
| Windows / macOS / 普通 Linux | `environment.yml` | 桥接、协议测试、离线示例 |
| Ubuntu 20.04 + NVIDIA GPU | ROS Noetic + `jev_obj/environment.sim.yml` | 完整仿真与评测 |
| Windows 11 + WSL2 + NVIDIA GPU | WSL2 Ubuntu 20.04 | 与原生 Ubuntu 基本相同 |
| 无桌面 GPU 服务器 | Ubuntu 20.04 + Habitat headless/EGL | 批量评测，不需要 RViz |
| 新款 GPU 或依赖冲突机器 | Habitat Python 3.9 与视觉 Python 3.10+ 分开 | 通过 HTTP 连接视觉服务 |

当前工程是 ROS 1 catkin 工作空间。Ubuntu 22.04/24.04 不要直接混装 Noetic 的 Ubuntu 20.04 二进制包，请使用 20.04 WSL、虚拟机或容器。

## 1. Jev 桥接环境

```bash
conda env create -f environment.yml
conda activate jev-obj-bridge
python -m pip install -e .
python -m unittest discover -s tests -v
```

离线接口检查：

```bash
python -m nav_jev_bridge.cli examples/snapshot.json --provider demo
python -m nav_jev_bridge.server --provider demo
curl http://127.0.0.1:8765/health
```

`demo` 只验证协议，不代表 Jev，也不能用于报告真实导航效果。

## 2. 本机 `.env` 配置

```bash
cp .env.example .env
```

| 字段 | 含义 |
|---|---|
| `JEV_OBJ_WSL_DISTRO` | Windows 下的 WSL 发行版名；原生 Linux 留空 |
| `JEV_OBJ_BRIDGE_PYTHON` | Python 3.10+ 桥接解释器绝对路径 |
| `JEV_OBJ_SIM_PYTHON` | Python 3.9 Habitat 解释器绝对路径 |
| `JEV_OBJ_BRIDGE_PORT` | 本地桥端口，默认 8766 |
| `JEV_OBJ_TRACE_DIR` | 请求审计目录，可写相对仓库根目录的路径 |
| `JEV_OPENROUTER_KEY_FILE` | 可选密钥文件路径；留空使用用户配置目录 |

工具只解析白名单里的 `NAME=VALUE`，不会执行 `.env` 中的 shell 表达式。值不要加引号，也不要上传 `.env`。

密钥单独保存到用户配置目录：

```bash
bash tools/setup-jev-key.sh
```

脚本隐藏输入并以权限 600 保存密钥；仓库不保存真实密钥。

## 3. ROS Noetic 与规划器

在 Ubuntu 20.04 安装 ROS Noetic 后：

```bash
sudo apt update
sudo apt install build-essential cmake git python3-catkin-pkg python3-empy \
  ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-pcl-ros \
  ros-noetic-rviz ros-noetic-cmake-modules libeigen3-dev libpcl-dev \
  libopencv-dev libompl-dev libarmadillo-dev libcurl4-openssl-dev
```

项目还需要 OSQP 0.6.3 与 OsqpEigen 0.8.1。安装后编译：

```bash
cd /path/to/jev-objnav/jev_obj
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release -j4
source devel/setup.bash
rospack find exploration_manager
```

`build/` 和 `devel/` 是本地产物，不上传。

## 4. Habitat 环境

```bash
conda env create -f jev_obj/environment.sim.yml
conda activate jev-obj-sim
python -c 'import habitat_sim, numpy; print(habitat_sim, numpy.__version__)'
```

Habitat-Lab 固定 v0.3.1：

```bash
cd /path/to/jev-objnav/jev_obj
mkdir -p .deps
git clone --branch v0.3.1 https://github.com/facebookresearch/habitat-lab.git .deps/habitat-lab
python -m pip install -e .deps/habitat-lab/habitat-lab
python -m pip install -e .deps/habitat-lab/habitat-baselines
python -m pip install -e .
```

无桌面服务器使用 Habitat-Sim headless/EGL。若新显卡与 Python 3.9 的 PyTorch 冲突，另建 Python 3.10+ 视觉环境，不要随意升级 Habitat 环境中的 Python 或 NumPy。

## 5. 数据、权重和视觉服务

数据不在仓库中。把已获授权的 HM3D 与 OVON 包放到仓库根目录，再运行：

```bash
python -m tools.extract_ovon_val
python -m tools.prepare_ovon_habitat031
python -m tools.prepare_ovon_pilot
```

权重放到 `jev_obj/data/`：`groundingdino_swint_ogc.pth`、`mobile_sam.pt`、`yolov7-e6e.pt`。源码依赖放到 `jev_obj/.deps/GroundingDINO/` 与 `jev_obj/.deps/yolov7/`。

四个真实视觉服务分别运行：

```bash
cd jev_obj
python -m vlm.detector.grounding_dino --port 12181
python -m vlm.itm.blip2itm --port 12182
python -m vlm.segmentor.sam --port 12183
python -m vlm.detector.yolov7 --port 12184
```

评测前运行 `python tools/check_real_perception_services.py`。所有服务必须显示真实 backend 且 `fallback=false`；空检测临时服务只能检查流程，不能用于有效测评。

## 6. 启动真实 Jev 与评测

```bash
bash tools/start-jev-region.sh --check
bash tools/start-jev-region.sh
```

健康检查应显示 `provider=openrouter`、`state_profile=region_graph` 和 schema `jev-region-graph/1.1`。另一个终端运行：

```bash
bash tools/run-region-eval.sh smoke 0
```

`smoke N` 中的 `N` 为零起始 episode 索引。脚本检查真实视觉服务、桥接实现、schema、ROS 参数和运行清单；日志与 trace 写入 `.runtime/`。

## 7. WSL 示例

在 PowerShell 只负责进入发行版，再在 Linux shell 内运行命令，避免复杂引号：

```powershell
wsl.exe -d Ubuntu-20.04
```

```bash
cd /path/to/jev-objnav
cp .env.example .env
# 编辑 .env 后：
bash tools/start-jev-region.sh --check
```

WSL 内先确认 `nvidia-smi` 与 PyTorch CUDA 可用。不要在 WSL 内安装 Linux 显示驱动，应使用支持 WSL 的 Windows NVIDIA 驱动。

