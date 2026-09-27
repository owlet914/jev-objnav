"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
import traceback
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence, SemanticObservation

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES, FINAL_RESULT
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub, itm_score_pub
    global observation_id, episode_key, latest_itm_state
    tmp = deepcopy(msg_observations)
    ros_time = rospy.Time.now()
    ros_pub.habitat_publish_ros_topic(tmp, ros_time=ros_time)
    publish_semantic_observation(
        itm_score_pub, ros_time, episode_key, observation_id, latest_itm_state
    )
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def publish_semantic_observation(publisher, stamp, episode, obs_id, state):
    msg = SemanticObservation()
    msg.header.stamp = stamp
    msg.header.frame_id = "world"
    msg.episode_id = episode
    msg.observation_id = obs_id
    msg.query_text = str(state.get("query_text", ""))
    msg.score_type = str(state.get("score_type", "blip2_itc_cosine_similarity"))
    msg.raw_score = float(state.get("raw_score", 0.0))
    msg.valid = bool(state.get("valid", False))
    msg.fallback = bool(state.get("fallback", False))
    msg.backend = str(state.get("backend", "not_observed_yet"))
    publisher.publish(msg)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d", "ovon"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2, mp3d or ovon (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global itm_score_pub
    global final_state, expl_result, observation_id, episode_key, latest_itm_state

    # Only MP3D needs this legacy category translation. OVON goal categories
    # are open-vocabulary strings and must reach the detector/ITM unchanged.
    category_to_coco = {}
    id_to_name = {}
    if cfg.habitat.dataset.split == "val" and "mp3d/v1" in cfg.habitat.dataset.data_path:
        with gzip.open(
            "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
        ) as f:
            val_data = json.load(f)
        category_to_coco = val_data.get("category_to_mp3d_category_id", {})
        id_to_name = {
            category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
            for idx, cat in enumerate(category_to_coco)
        }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test
    eval_episode_limit = int(getattr(cfg, "eval_episode_limit", -1))
    if eval_episode_limit == 0 or eval_episode_limit < -1:
        raise ValueError("eval_episode_limit must be -1 or a positive integer")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    audit = None
    if os.environ.get("JEV_REGION_RUN_DIR"):
        from region_evaluation_audit import RegionEvaluationAudit
        audit = RegionEvaluationAudit(cfg, env)
    print("Environment creation successful")
    turn_angle_deg = float(cfg.habitat.simulator.turn_angle)
    if not np.isfinite(turn_angle_deg) or turn_angle_deg <= 0.0 or turn_angle_deg > 180.0:
        raise ValueError(f"Invalid Habitat turn_angle: {turn_angle_deg}")
    rospy.set_param("/jev_obj/jev/observation_turn_angle_deg", turn_angle_deg)
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    remaining_episodes = number_of_episodes - num_total
    episodes_to_run = 1 if flag_once else remaining_episodes
    if not flag_once and eval_episode_limit > 0:
        episodes_to_run = min(episodes_to_run, eval_episode_limit)
    pbar = tqdm.tqdm(total=episodes_to_run)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher(
        "/blip2/semantic_observation", SemanticObservation, queue_size=10
    )
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)

    for epi in range(episodes_to_run):
        # Publish progress information
        publish_int32_array(progress_pub, [epi, episodes_to_run])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        final_state = 0
        expl_result = 0
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0
        observation_id = 0

        camera_pitch = 0.0
        observations = env.reset()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        # Context for the optional high-level Jev planner. Room is an LLM prior,
        # not a room observation from the map, so the bridge keeps it separate.
        episode_info = env.current_episode.info or {}
        source_episode_id = episode_info.get(
            "ovon_episode_id", env.current_episode.episode_id
        )
        episode_key = f"{os.path.basename(env.current_episode.scene_id)}:{source_episode_id}"
        if audit:
            audit.check_bridge()
            audit.event("episode_start",episode_id=episode_key,target=label,episode_index=env_num_once if flag_once else num_total)
        rospy.set_param("/jev_obj/jev/accepted_request_id", "")
        rospy.set_param("/jev_obj/jev/last_error", "")
        latest_itm_state = {
            "raw_score": 0.0,
            "valid": False,
            "fallback": False,
            "backend": "not_observed_yet",
            "query_text": "",
            "score_type": "blip2_itc_cosine_similarity",
        }
        rospy.set_param("/jev_obj/jev/episode_id", episode_key)
        rospy.set_param("/jev_obj/jev/target_object", label)
        rospy.set_param("/jev_obj/jev/room_prior", room)
        rospy.set_param("/jev_obj/jev/related_objects", [str(item) for item in llm_answer])
        rospy.set_param("/jev_obj/jev/latest_itm_score", -1.0)
        rospy.set_param("/jev_obj/jev/perception/itm_available", False)
        rospy.set_param("/jev_obj/jev/perception/itm_fallback", False)
        rospy.set_param("/jev_obj/jev/perception/itm_backend", "not_observed_yet")
        rospy.set_param("/jev_obj/jev/perception/object_detection_available", False)
        rospy.set_param("/jev_obj/jev/perception/object_detection_fallback", False)
        rospy.set_param("/jev_obj/jev/perception/object_detection_backends", [])
        rospy.set_param("/jev_obj/jev/perception/object_detection_count", 0)
        rospy.set_param("/jev_obj/jev/navigation/step_count", 0)
        rospy.set_param("/jev_obj/jev/navigation/latest_collision", False)
        rospy.set_param("/jev_obj/jev/navigation/collision_count", 0)
        rospy.set_param(
            "/jev_obj/jev/target_subcategories",
            [str(item) for item in episode_info.get("children_object_categories", [])],
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            accepted_id = rospy.get_param("/jev_obj/jev/accepted_request_id", "")
            if audit:
                audit.event("action_dispatch",episode_id=episode_key,observation_id=observation_id,request_id=accepted_id,
                    candidate_id=rospy.get_param("/jev_obj/jev/accepted_candidate_id", ""),
                    action_origin=rospy.get_param("/jev_obj/jev/navigation/last_action_origin", "unknown"),
                    mode=rospy.get_param("/jev_obj/jev/accepted_mode", ""),action=int(action))
            observations = env.step(action)
            observation_id += 1

            # Calculate ITM cosine similarity score
            cosine, itm_status = get_itm_message_cosine(
                observations["rgb"], label, room, return_metadata=True
            )
            print(f"Target related room: {room}")
            print(
                f"ITM cosine similarity: {cosine:.3f}; "
                f"available={itm_status['available']}; fallback={itm_status['fallback']}"
            )

            rospy.set_param(
                "/jev_obj/jev/perception/itm_available", bool(itm_status["available"])
            )
            rospy.set_param(
                "/jev_obj/jev/perception/itm_fallback", bool(itm_status["fallback"])
            )
            rospy.set_param(
                "/jev_obj/jev/perception/itm_backend", str(itm_status["backend"])
            )
            latest_itm_state = {
                **itm_status,
                "raw_score": float(cosine),
                "valid": bool(itm_status["available"] and not itm_status["fallback"]),
            }
            rospy.set_param("/jev_obj/jev/perception/itm_observation_id", int(observation_id))
            rospy.set_param("/jev_obj/jev/perception/itm_query_text", str(itm_status["query_text"]))
            rospy.set_param("/jev_obj/jev/perception/itm_score_type", str(itm_status["score_type"]))
            if latest_itm_state["valid"]:
                rospy.set_param("/jev_obj/jev/latest_itm_score", float(cosine))
            else:
                rospy.set_param("/jev_obj/jev/latest_itm_score", -1.0)

            # Detect objects in the current observation
            (
                observations["rgb"],
                score_list,
                object_masks_list,
                label_list,
                detection_status,
            ) = get_object(
                label, observations["rgb"], detector_cfg, llm_answer,
                return_metadata=True,
            )
            rospy.set_param(
                "/jev_obj/jev/perception/object_detection_available",
                bool(detection_status["target_available"]),
            )
            rospy.set_param(
                "/jev_obj/jev/perception/object_detection_fallback",
                bool(detection_status["fallback"]),
            )
            rospy.set_param(
                "/jev_obj/jev/perception/object_detection_backends",
                [str(item["backend"]) for item in detection_status["detectors"]],
            )
            rospy.set_param(
                "/jev_obj/jev/perception/object_detection_count",
                int(sum(item["raw_detection_count"] for item in detection_status["detectors"])),
            )
            rospy.set_param("/jev_obj/jev/perception/detection_observation_id", int(observation_id))
            rospy.set_param("/jev_obj/jev/perception/target_detection_available", bool(detection_status["target_available"]))
            rospy.set_param("/jev_obj/jev/perception/related_detection_available", bool(detection_status["related_available"]))
            rospy.set_param("/jev_obj/jev/perception/target_match_count", int(detection_status["target_match_count"]))
            rospy.set_param("/jev_obj/jev/perception/related_match_count", int(detection_status["related_match_count"]))
            rospy.set_param("/jev_obj/jev/perception/valid_mask_count", int(detection_status["valid_mask_count"]))
            rospy.set_param(
                "/jev_obj/jev/perception/detector_status_json",
                json.dumps(detection_status["detectors"], ensure_ascii=False),
            )

            if audit and (not itm_status["available"] or itm_status["fallback"] or
                    not detection_status["target_available"] or not detection_status["related_available"] or detection_status["fallback"]):
                audit.result(episode_id=episode_key,target=label,steps=count_steps,success=0,spl=0.0,technical_failure=True,reason="PERCEPTION_UNAVAILABLE",completed=False)
                raise RuntimeError("PERCEPTION_UNAVAILABLE: stop batch")

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            observation_stamp = rospy.Time.now()
            ros_pub.habitat_publish_ros_topic(msg_observations, ros_time=observation_stamp)
            publish_semantic_observation(
                itm_score_pub, observation_stamp, episode_key, observation_id, latest_itm_state
            )

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_msg.header.stamp = observation_stamp
            cld_with_score_msg.header.frame_id = "world"
            cld_with_score_msg.episode_id = episode_key
            cld_with_score_msg.observation_id = observation_id
            cld_with_score_msg.detector_names = [str(item["name"]) for item in detection_status["detectors"]]
            cld_with_score_msg.detector_backends = [str(item["backend"]) for item in detection_status["detectors"]]
            cld_with_score_msg.detector_available = [bool(item["available"]) for item in detection_status["detectors"]]
            cld_with_score_msg.detector_fallback = [bool(item["fallback"]) for item in detection_status["detectors"]]
            cld_with_score_msg.detector_target_coverage = [bool(item["target_coverage"]) for item in detection_status["detectors"]]
            cld_with_score_msg.detector_requested_classes = [json.dumps(item["requested_classes"], ensure_ascii=False) for item in detection_status["detectors"]]
            cld_with_score_msg.raw_detection_counts = [int(item["raw_detection_count"]) for item in detection_status["detectors"]]
            cld_with_score_msg.target_match_counts = [int(item["target_match_count"]) for item in detection_status["detectors"]]
            cld_with_score_msg.related_match_counts = [int(item["related_match_count"]) for item in detection_status["detectors"]]
            cld_with_score_msg.valid_mask_counts = [int(item["valid_mask_count"]) for item in detection_status["detectors"]]
            cld_with_score_msg.label_detection_valid = [
                bool(item) for item in detection_status["label_detection_valid"]
            ]
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            collision_metrics = info.get("collisions", {})
            rospy.set_param("/jev_obj/jev/navigation/step_count", int(count_steps))
            rospy.set_param(
                "/jev_obj/jev/navigation/latest_collision",
                bool(collision_metrics.get("is_collision", False)),
            )
            rospy.set_param(
                "/jev_obj/jev/navigation/collision_count",
                int(collision_metrics.get("count", 0)),
            )
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            if audit:
                audit.event("action_completion",episode_id=episode_key,request_id=accepted_id,
                    next_observation_id=observation_id,steps=count_steps,collision=bool(collision_metrics.get("is_collision",False)))
            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if final_state == FINAL_RESULT.JEV_FAILURE:
            result_text = "jev decision failure"
            success = 0
            spl = 0.0
            soft_spl = 0.0
        elif success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if audit:
            technical_failure = final_state == FINAL_RESULT.JEV_FAILURE
            audit.result(episode_id=episode_key,target=label,steps=count_steps,success=float(success),spl=float(spl),
                reason=rospy.get_param("/jev_obj/jev/last_error", "") or result_text,technical_failure=technical_failure,completed=True)
            if technical_failure:
                raise RuntimeError("JEV_TECHNICAL_FAILURE: stop batch after recording failure")
        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        if num_total < number_of_episodes:
            env.current_episode = next(env.episode_iterator)
            rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        traceback.print_exc()
        rospy.signal_shutdown("Shutdown due to error")
        raise
