import copy
import unittest

from nav_jev_bridge.bridge import decide, prepare_decision
from nav_jev_bridge.region_graph import RegionGraphError, validate_region_graph_state


def region_snapshot():
    state = {
        "schema_version": "jev-region-graph/1.1",
        "representation": {"kind": "region_graph", "lossy": True, "frame_id": "world"},
        "identity": {
            "episode_id": "episode-1", "request_id": "episode-1:obs-4:req-1",
            "observation_id": 4, "map_revision": 5, "region_graph_revision": 2,
        },
        "task": {"target_object": "chair", "room_prior": None, "related_objects": []},
        "robot": {"region_id": "R0001", "region_assignment": "exact_free_cell",
                  "pose": {"x": 0.5, "y": 0.5, "yaw": 0.0}},
        "perception": {},
        "jev_obj_policy": {
            "mode": "hybrid", "mode_id": 2,
            "decision_order": ["high_confidence_object", "active_frontier_policy"],
            "semantic_thresholds": {
                "std_dev": 0.03, "max_to_mean": 1.2,
                "max_to_mean_percentage": 0.95,
            },
            "active_frontier_statistics": {
                "total_count": 1, "reachable_count": 1, "mean": 0.2,
                "std_dev": 0.0, "maximum": 0.2, "max_to_mean": 1.0,
                "zero_mean_ratio_definition": 1.0,
                "adaptive_max_to_mean_threshold": 1.2,
            },
            "dormant_frontier_statistics": {
                "total_count": 0, "reachable_count": 0, "mean": 0.0,
                "std_dev": 0.0, "maximum": 0.0, "max_to_mean": 1.0,
                "zero_mean_ratio_definition": 1.0,
                "adaptive_max_to_mean_threshold": 1.2,
            },
            "high_semantic_subset": [
                {"frontier_id": "frontier-2", "selected": False,
                 "value_to_mean": 1.0},
            ],
            "hybrid_branch": "closest",
            "available_stages": ["high_confidence_object", "active_frontier_policy"],
            "earliest_available_stage": "high_confidence_object",
            "earliest_available_priority": 1,
            "interpretation": "reference gate",
        },
        "object_confirmation": {
            "status": "idle", "pending_candidate_id": None,
            "pending_target_id": None, "start_observation_id": None,
            "last_observation_id": None, "attempts": 0,
            "valid_fresh_hits": 0, "required_fresh_hits": 2,
            "max_attempts": 4, "last_outcome": "idle",
            "blocked_candidate_id": None, "blocked_target_id": None,
            "blocked_until_observation_id": None,
            "stop_rule": "fresh multi-view confirmation required",
        },
        "regions": [
            {"id": "R0001", "bbox_xy": {"min": [0.0, 0.0], "max": [3.0, 3.0]},
             "anchor_xy": [0.5, 0.5], "free_area_m2": 2.0, "visited": True,
             "semantic": {"valid": True, "mean": -0.1, "p90": 0.0, "max": 0.2,
                          "valid_area_fraction": 0.5}, "objects": [], "frontiers": []},
            {"id": "R0002", "bbox_xy": {"min": [3.0, 0.0], "max": [6.0, 3.0]},
             "anchor_xy": [3.5, 0.5], "free_area_m2": 2.0, "visited": False,
             "semantic": {"valid": False, "mean": None, "p90": None, "max": None,
                          "valid_area_fraction": 0.0},
             "objects": [{"id": "object-7", "target_confidence": 0.7,
                            "selectable": True}],
             "frontiers": [{"id": "frontier-2", "state": "active",
                              "semantic_valid": False, "semantic_score": None,
                              "selectable": True}]},
        ],
        "connections": [{"id": "E1", "from": "R0001", "to": "R0002",
                         "bidirectional": True, "status": "verified_reachable",
                         "verified_map_revision": 5}],
        "options": [
            {"id": "explore:frontier-2-goal", "action_type": "explore",
             "exploration_kind": "frontier", "target_id": "frontier-2",
             "region_id": "R0002", "distance_m": 2.5,
             "route_region_ids": ["R0001", "R0002"],
             "candidate_id": "frontier-2-goal", "local_path_verified": True,
             "repeated_selection_count": 0,
             "source_stage": "active_frontier_policy", "source_priority": 3,
             "original_policy_stage_active": False, "safety_mode": "normal",
             "semantic_score": None, "information_gain": None},
            {"id": "approach:object-7-goal", "action_type": "approach_object",
             "target_id": "object-7", "region_id": "R0002", "distance_m": 2.0,
             "route_region_ids": ["R0001", "R0002"],
             "candidate_id": "object-7-goal", "local_path_verified": True,
             "repeated_selection_count": 0,
             "source_stage": "high_confidence_object", "source_priority": 1,
             "original_policy_stage_active": True, "safety_mode": "normal",
             "semantic_score": 0.7, "information_gain": None},
        ],
        "history": {},
        "coverage": {
            "raw_map_complete": False, "high_level_evidence_complete": False,
            "decision_evidence_complete": True,
            "summary_contract_valid": True, "regions_exported": 2,
            "objects_exported": 1, "frontiers_exported": 1,
            "executable_candidates_exported": 2, "omitted_entities": 0,
        },
        "unassigned_entities": {"objects": [], "frontiers": []},
        "selection_parameters": {
            "min_target_confidence": 0.5,
            "min_target_observations": 2,
            "min_independent_viewpoints": 2,
            "min_viewpoint_baseline_m": 0.35,
            "eligibility_evaluated_locally_at_full_precision": True,
            "class_function_score_definition": "int(points * confidence)",
        },
    }
    for key,value in list(state["coverage"].items()):
        if key.endswith("_exported"): state["coverage"][key.replace("_exported","_total")] = value
    candidates = [
        {"id": "frontier-2-goal", "kind": "explore_frontier", "target_id": "frontier-2",
         "goal_pose": {"x": 4.0, "y": 0.5, "yaw": 0.0},
         "path": {"reachable": True, "collision_free": True, "distance_m": 2.5}},
        {"id": "object-7-goal", "kind": "approach_object", "target_id": "object-7",
         "goal_pose": {"x": 3.8, "y": 0.5, "yaw": 0.0},
         "path": {"reachable": True, "collision_free": True, "distance_m": 2.0}},
    ]
    return {"region_graph": state, "candidates": candidates}


class ChoiceProvider:
    def choose_request(self, payload):
        self.payload = payload
        return "approach:object-7-goal", 0.8, {
            "approach:object-7-goal": 0.8, "explore:frontier-2-goal": 0.2,
        }


class MetadataChoiceProvider:
    def choose_request_with_metadata(self, payload):
        return "approach:object-7-goal", 0.8, {
            "approach:object-7-goal": 0.8, "explore:frontier-2-goal": 0.2,
        }, {"actual_model": "typesafe/jev-test", "generation_id": "gen-1",
            "usage": {"input_tokens": 123}, "raw_response": {"id": "gen-1"}}


class RegionGraphContractTests(unittest.TestCase):
    def test_projection_is_lossy_and_maps_option_to_local_candidate(self):
        snapshot = region_snapshot()
        snapshot["region_graph"]["regions"][1]["objects"][0].update({
            "target_observations": 3,
            "target_unique_viewpoints": 1,
            "target_viewpoint_baseline_m": 0.0,
            "target_viewpoint_bearing_span_rad": 0.0,
            "extent_xyz_m": [0.8, 0.5, 1.0],
            "stored_best_label": "chair",
            "computed_best_label": "chair",
            "class_evidence": [
                {"label": "chair", "fused_confidence": 0.7,
                 "observation_count": 3, "integer_function_score": 2},
                {"label": "stool", "fused_confidence": 0.5,
                 "observation_count": 2, "integer_function_score": 1},
                {"label": "sofa", "fused_confidence": 0.4,
                 "observation_count": 2, "integer_function_score": 1},
                {"label": "table", "fused_confidence": 0.2,
                 "observation_count": 1, "integer_function_score": 0},
                {"label": "cabinet", "fused_confidence": 0.1,
                 "observation_count": 1, "integer_function_score": 0},
            ],
            "selection_status": "current_frame_high_confidence_target_with_verified_path",
            "high_confidence_eligible": True,
            "updated_in_current_observation": True,
            "observations_since_seen": 0,
            "post_arrival_confirmation_status": "not_arrived_current_frame_supported",
        })
        prepared = prepare_decision(snapshot, "region_graph")
        self.assertEqual(prepared.state["schema_version"], "jev-region-graph/1.1")
        self.assertTrue(all("path" not in option and "points" not in option
                            for option in prepared.state["options"]))
        self.assertEqual(
            [option["candidate_id"] for option in prepared.state["options"]],
            ["frontier-2-goal", "object-7-goal"],
        )
        compact_object = prepared.state["regions"][1]["objects"][0]
        self.assertEqual(len(compact_object["class_evidence"]), 2)
        self.assertEqual(compact_object["class_evidence_omitted_count"], 3)
        provider = ChoiceProvider()
        result = decide(snapshot, provider, state_profile="region_graph")
        self.assertEqual(result["candidate_id"], "object-7-goal")
        self.assertEqual(result["selected_option_id"], "approach:object-7-goal")
        self.assertEqual(result["region_graph_revision"], 2)
        criterion = provider.payload["questions"]["next_goal"]["criteria"][
            "approach:object-7-goal"
        ]
        self.assertIn("frames=3", criterion)
        self.assertIn("physical_views=1", criterion)
        self.assertIn("baseline=0.0m", criterion)
        self.assertIn("fresh=True", criterion)
        self.assertIn("approach is not success", criterion)
        self.assertEqual(compact_object["extent_xyz_m"], [0.8, 0.5, 1.0])
        self.assertEqual(
            compact_object["selection_status"],
            "current_frame_high_confidence_target_with_verified_path",
        )
        self.assertEqual(
            compact_object["post_arrival_confirmation_status"],
            "not_arrived_current_frame_supported",
        )
        self.assertEqual(
            compact_object["confidence_context"]["tier"],
            "supported_above_gate",
        )
        self.assertAlmostEqual(
            compact_object["confidence_context"]["delta_to_gate"], 0.2
        )
        self.assertEqual(
            prepared.state["semantic_alignment"]["queried_class_comparison"]["scope"],
            "target label plus task-provided related labels only",
        )
        self.assertIn(
            "too_few_physical_camera_positions",
            compact_object["verification_need"]["reasons"],
        )
        self.assertIn(
            "viewpoint_translation_baseline_too_small",
            compact_object["verification_need"]["reasons"],
        )
        self.assertEqual(
            prepared.state["semantic_alignment"]["object_confidence"]["operational_gate"],
            0.5,
        )
        self.assertIn("not probabilities", prepared.model_request["questions"]["next_goal"]["instructions"])

    def test_raw_detail_leak_is_rejected(self):
        state = region_snapshot()["region_graph"]
        state["regions"][0]["cells"] = [[1, 2]]
        with self.assertRaisesRegex(RegionGraphError, "forbidden"):
            validate_region_graph_state(state)

    def test_targeted_observation_exposes_focus_and_evidence_goal(self):
        snapshot = region_snapshot()
        state = snapshot["region_graph"]
        state["observation_actions"] = [{
            "id": "inspect-object-7-left", "region_id": "R0001",
            "direction": "left", "rotation_rad": 0.52, "expected_yaw": 0.52,
            "selectable": True, "focus_kind": "object",
            "focus_target_id": "object-7", "focus_region_id": "R0002",
            "focus_bearing_rad": 0.7,
            "evidence_gaps": ["only_narrowly_above_local_gate"],
            "purpose": "inspect the leading object hypothesis",
            "expected_effect": "one fresh synchronized observation",
        }]
        state["options"].append({
            "id": "explore:inspect-object-7-left-goal", "action_type": "explore",
            "exploration_kind": "observe_rotation", "target_id": "inspect-object-7-left",
            "region_id": "R0001", "distance_m": 0.0,
            "route_region_ids": ["R0001"],
            "candidate_id": "inspect-object-7-left-goal", "local_path_verified": True,
            "repeated_selection_count": 0, "source_stage": "targeted_object_verification",
            "source_priority": 2, "original_policy_stage_active": False,
            "safety_mode": "in_place_targeted", "semantic_score": 0.7,
            "information_gain": None,
        })
        snapshot["candidates"].append({
            "id": "inspect-object-7-left-goal", "kind": "observe_rotation",
            "target_id": "inspect-object-7-left",
            "goal_pose": {"x": 0.5, "y": 0.5, "yaw": 0.52},
            "path": {"reachable": True, "collision_free": True, "distance_m": 0.0},
        })
        state["coverage"].update(
            executable_candidates_total=3, executable_candidates_exported=3
        )
        prepared = prepare_decision(snapshot, "region_graph")
        option = next(
            item for item in prepared.state["options"]
            if item["candidate_id"] == "inspect-object-7-left-goal"
        )
        action = next(
            item for item in prepared.state["observation_actions"]
            if item["id"] == option["target_id"]
        )
        self.assertEqual(action["focus_kind"], "object")
        self.assertEqual(action["focus_target_id"], "object-7")
        self.assertIn(
            "only_narrowly_above_local_gate",
            action["evidence_gaps"],
        )
        self.assertIn(
            "focus=object:object-7",
            prepared.options["explore:inspect-object-7-left-goal"],
        )

    def test_decision_local_projection_retains_every_option_but_summarizes_remote_regions(self):
        snapshot = region_snapshot()
        state = snapshot["region_graph"]
        state["regions"].append({
            "id": "R0099", "bbox_xy": {"min": [20.0, 20.0], "max": [23.0, 23.0]},
            "anchor_xy": [21.0, 21.0], "free_area_m2": 3.0, "visited": False,
            "semantic": {"valid": False, "mean": None, "p90": None, "max": None,
                         "valid_area_fraction": 0.0},
            "objects": [],
            "frontiers": [],
        })
        state["coverage"].update(
            regions_total=3, regions_exported=3,
        )
        prepared = prepare_decision(snapshot, "region_graph")
        self.assertEqual(len(prepared.options), 2)
        self.assertNotIn("R0099", {row["id"] for row in prepared.state["regions"]})
        self.assertTrue(prepared.state["projection_summary"]["all_safe_options_retained"])
        self.assertEqual(prepared.state["projection_summary"]["regions_summarized"], 1)
        self.assertEqual(prepared.state["coverage"]["regions_total"], 3)
        self.assertEqual(prepared.state["coverage"]["regions_exported"], 2)

    def test_route_and_coverage_mismatch_are_rejected(self):
        snapshot = region_snapshot()
        broken = copy.deepcopy(snapshot)
        broken["region_graph"]["options"][0]["route_region_ids"] = ["R0002"]
        with self.assertRaises(ValueError):
            prepare_decision(broken, "region_graph")
        broken = copy.deepcopy(snapshot)
        broken["region_graph"]["coverage"]["frontiers_exported"] = 0
        with self.assertRaises(ValueError):
            prepare_decision(broken, "region_graph")
        broken = copy.deepcopy(snapshot)
        broken["region_graph"]["connections"] = []
        with self.assertRaisesRegex(ValueError, "unverified edge"):
            prepare_decision(broken, "region_graph")

    def test_missing_local_candidate_is_rejected(self):
        snapshot = region_snapshot()
        snapshot["candidates"] = snapshot["candidates"][:1]
        with self.assertRaisesRegex(ValueError, "missing local candidates"):
            prepare_decision(snapshot, "region_graph")

    def test_nonfinite_and_mode_target_mismatch_are_rejected(self):
        state = region_snapshot()["region_graph"]
        state["regions"][0]["semantic"]["mean"] = float("nan")
        with self.assertRaisesRegex(RegionGraphError, "finite"):
            validate_region_graph_state(state)
        state = region_snapshot()["region_graph"]
        state["options"][0]["action_type"] = "approach_object"
        with self.assertRaisesRegex(RegionGraphError, "not an object"):
            validate_region_graph_state(state)

    def test_provider_metadata_is_preserved_for_trace(self):
        _, trace = decide(
            region_snapshot(), MetadataChoiceProvider(), state_profile="region_graph",
            return_trace=True,
        )
        self.assertEqual(trace["provider_metadata"]["generation_id"], "gen-1")
        self.assertEqual(trace["provider_metadata"]["raw_response"], {"id": "gen-1"})

    def test_recent_observations_are_validated_as_a_list_of_rows(self):
        snapshot = region_snapshot()
        snapshot["region_graph"]["perception"]["recent_observations"] = [{
            "observation_id": 4,
            "image_text_match_valid": True,
            "image_text_match_raw_score": 0.21,
            "target_match_count": 1,
            "valid_mask_count": 1,
        }]
        prepared = prepare_decision(snapshot, "region_graph")
        self.assertEqual(
            prepared.state["perception"]["recent_observations"][0]["observation_id"],
            4,
        )

    def test_rejects_unsafe_mismatched_and_duplicate_local_registry(self):
        for mutation in ('collision','mode','duplicate','pose'):
            snapshot=region_snapshot()
            if mutation=='collision': snapshot['candidates'][1]['path']['collision_free']=False
            if mutation=='mode': snapshot['candidates'][1]['kind']='explore_frontier'
            if mutation=='duplicate': snapshot['candidates'].append(copy.deepcopy(snapshot['candidates'][1]))
            if mutation=='pose': snapshot['candidates'][1]['goal_pose']['x']=float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_decision(snapshot,'region_graph')

    def test_rejects_stale_edge_wrong_source_counts_and_extra_fields(self):
        for mutation in ('edge','total','unknown','invalid_semantic','unsupported_frontier'):
            snapshot=region_snapshot(); state=snapshot['region_graph']
            if mutation=='edge': state['connections'][0]['verified_map_revision']=0
            if mutation=='total': state['coverage']['objects_total']=99
            if mutation=='unknown': state['ground_truth']={'position':[1,2]}
            if mutation=='invalid_semantic': state['regions'][1]['semantic']['mean']=0.9
            if mutation=='unsupported_frontier': state['regions'][1]['frontiers'][0].update(semantic_valid=True,semantic_score=0,fusion_weight=0)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_decision(snapshot,'region_graph')

    def test_retains_all_safe_approaches_across_regions(self):
        snapshot=region_snapshot(); state=snapshot['region_graph']
        candidate=copy.deepcopy(snapshot['candidates'][1]); candidate['id']='object-7-alt'
        candidate['goal_pose']['x']=1.0
        snapshot['candidates'].append(candidate)
        option=copy.deepcopy(state['options'][1])
        option.update(id='approach:object-7-alt',candidate_id=candidate['id'],region_id='R0001',route_region_ids=['R0001'])
        state['options'].append(option)
        state['coverage'].update(executable_candidates_total=3,executable_candidates_exported=3)
        prepared=prepare_decision(snapshot,'region_graph')
        self.assertEqual(len(prepared.options),3)
        self.assertEqual(prepared.option_to_candidate[option['id']],candidate['id'])

    def test_summary_valid_is_set_only_after_validation(self):
        snapshot=region_snapshot(); snapshot['region_graph']['coverage']['summary_contract_valid']=False
        prepared=prepare_decision(snapshot,'region_graph')
        self.assertTrue(prepared.state['coverage']['summary_contract_valid'])
        self.assertFalse(snapshot['region_graph']['coverage']['summary_contract_valid'])


if __name__ == "__main__":
    unittest.main()
