"""Reset and step one pilot OVON episode in Habitat-Lab 0.3.1.

This checks the real ObjectNav environment and rendering. The action is a
single forward step, not an Jev Obj/Jev policy or a success-rate trial.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import habitat
from habitat.config.default import get_config
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("pilot", "dev"), default="pilot")
    parser.add_argument("--output", type=Path, default=Path("examples/ovon_episode_step.png"))
    args = parser.parse_args()
    config = get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=[
            "habitat.dataset.type=ObjectNav-v1",
            f"habitat.dataset.split={args.split}",
            f"habitat.dataset.data_path=data/datasets/objectnav/ovon_small/{args.split}/{args.split}.json.gz",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=-1",
            "habitat.task.measurements.success.success_distance=0.25",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=256",
            "habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=256",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=256",
            "habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=256",
        ],
    )
    with habitat.Env(config=config) as env:
        initial = env.reset()
        if "rgb" not in initial:
            raise RuntimeError("Habitat did not return an RGB observation")
        episode = env.current_episode
        observation = env.step({"action": "move_forward"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observation["rgb"][..., :3]).save(args.output)
        metrics = env.get_metrics()
        print(json.dumps({
            "split": args.split,
            "episodes_available": env.number_of_episodes,
            "scene": Path(episode.scene_id).name,
            "target": episode.object_category,
            "rgb_shape": list(observation["rgb"].shape),
            "step_completed": True,
            "metric_names": sorted(metrics),
            "image": str(args.output),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
