"""Verify pilot Habitat observations can be serialized as ROS Image messages."""
from __future__ import annotations

import json
from pathlib import Path

import habitat
from habitat.config.default import get_config
from habitat2ros.habitat_publisher import ROSPublisher


def main() -> None:
    config = get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml",
        overrides=[
            "habitat.dataset.type=ObjectNav-v1",
            "habitat.dataset.split=pilot",
            "habitat.dataset.data_path=data/datasets/objectnav/ovon_small/pilot/pilot.json.gz",
            "habitat.simulator.habitat_sim_v0.gpu_device_id=-1",
        ],
    )
    with habitat.Env(config=config) as env:
        observation = env.reset()
        rgb = ROSPublisher._image_message(None, observation["rgb"], "rgb8")
        depth = ROSPublisher._image_message(None, observation["depth"], "32FC1")
        print(json.dumps({
            "scene": Path(env.current_episode.scene_id).name,
            "target": env.current_episode.object_category,
            "rgb": [rgb.width, rgb.height, rgb.step, rgb.encoding, len(rgb.data)],
            "depth": [depth.width, depth.height, depth.step, depth.encoding, len(depth.data)],
        }))


if __name__ == "__main__":
    main()
