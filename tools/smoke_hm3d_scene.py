"""Load one HM3D OVON pilot scene in Habitat-Sim and render its start view.

This is a scene-load check, not an Jev Obj/Jev navigation evaluation. It reads
only an episode's start pose; goal coordinates remain reserved for scoring.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import habitat_sim
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_from_coeffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path,
        default=Path("jev_obj/data/scene_datasets/hm3d/val/00869-MHPLjHsuG27/MHPLjHsuG27.basis.glb"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("jev_obj/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"),
    )
    parser.add_argument(
        "--episodes", type=Path,
        default=Path("jev_obj/data/datasets/objectnav/ovon_small/pilot/content/MHPLjHsuG27.json.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("examples/ovon_scene_smoke.png"))
    parser.add_argument("--gpu-id", type=int, default=-1)
    args = parser.parse_args()

    with gzip.open(args.episodes, "rt", encoding="utf-8") as stream:
        episode = json.load(stream)["episodes"][0]
    start = episode["start_position"]

    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = str(args.scene.resolve())
    simulator.scene_dataset_config_file = str(args.config.resolve())
    simulator.gpu_device_id = args.gpu_id
    color = habitat_sim.CameraSensorSpec()
    color.uuid = "color_sensor"
    color.sensor_type = habitat_sim.SensorType.COLOR
    color.resolution = [256, 256]
    color.position = [0.0, 1.5, 0.0]
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [color]

    with habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent_config])) as sim:
        navmesh = args.scene.with_suffix(".navmesh")
        navmesh_loaded = sim.pathfinder.is_loaded or sim.pathfinder.load_nav_mesh(str(navmesh))
        agent = sim.initialize_agent(0)
        state = agent.get_state()
        state.position = np.asarray(start, dtype=np.float32)
        state.rotation = quat_from_coeffs(episode["start_rotation"])
        agent.set_state(state)
        observations = sim.get_sensor_observations()
        rgb = observations["color_sensor"][..., :3]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        print(json.dumps({
            "scene": str(args.scene),
            "navmesh_loaded": bool(navmesh_loaded),
            "start_navigable": bool(sim.pathfinder.is_navigable(np.asarray(start))),
            "navigable_area_m2": float(sim.pathfinder.navigable_area),
            "image": str(args.output),
            "image_size": list(rgb.shape[:2]),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
