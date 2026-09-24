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
    workspace_lib = (Path(__file__).resolve().parents[1] / "jev_obj" / "devel" / "lib").as_posix()
    ros_paths = [path for path in sys.path if path.startswith(("/opt/ros/noetic/", "/usr/lib/python3/dist-packages", workspace_lib))]
    for path in ros_paths:
        sys.path.remove(path)
    sys.path.extend(ros_paths)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
