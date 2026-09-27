"""Run a script in the Focal Conda environment with ROS paths after Conda packages.

ROS Noetic's Python 3.8 system NumPy must not shadow Conda's Python 3.9 NumPy.
Use after sourcing /opt/ros/noetic/setup.bash and jev_obj/devel/setup.bash.
"""
import runpy
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_focal_python.py SCRIPT [ARGS...]")
    workspace = Path(__file__).resolve().parents[1] / "jev_obj"
    workspace_lib = (workspace / "devel" / "lib").as_posix()
    ros_paths = [path for path in sys.path if path.startswith(("/opt/ros/noetic/", "/usr/lib/python3/dist-packages", workspace_lib))]
    for path in ros_paths:
        sys.path.remove(path)
    sys.path.extend(ros_paths)
    # ROS Noetic helper modules live in the Focal system dist-packages.
    system_ros_packages = "/usr/lib/python3/dist-packages"
    if system_ros_packages not in sys.path:
        sys.path.append(system_ros_packages)
    script = sys.argv[1]
    # run_path does not add the target script directory to sys.path.
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(Path(script).resolve().parent))
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
