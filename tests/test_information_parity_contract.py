import copy
import unittest

from nav_jev_bridge.bridge import (
    ProviderError,
    SnapshotError,
    build_model_state,
    decide,
    prepare_decision,
    validate_snapshot,
)


def schema2_snapshot(object_count=70, active_count=8, dormant_count=7):
    objects = [
        {
            "id": f"object-{index}", "position": {"x": float(index), "y": 0.0},
            "selection": {"relaxed_eligible": False},
            "normal_path_search": {"geometry_available": True, "path_points": []},
        }
        for index in range(object_count)
    ]
    raw_neighborhood = [
        {"dx": dx, "dy": dy, "in_map": True, "semantic_raw": 0.0}
        for dx in range(-2, 3) for dy in range(-2, 3)
    ]
    frontiers = [
        {
            "id": f"frontier-{index}", "position": {"x": float(index), "y": 1.0},
            "state": "active", "planning_status": "reach_end", "path": {},
            "semantic_features": {"raw_neighborhood": copy.deepcopy(raw_neighborhood)},
        }
        for index in range(active_count)
    ] + [
        {
            "id": f"frontier-{active_count + index}",
            "position": {"x": float(index), "y": 2.0}, "state": "dormant",
            "planning_status": "reach_end", "path": {},
            "semantic_features": {"raw_neighborhood": copy.deepcopy(raw_neighborhood)},
        }
        for index in range(dormant_count)
    ]
    candidate_evidence = [
        {"target_entity_id": item["id"], "search_result": "reach_end"}
        for item in frontiers
    ]

    def graph(start, count):
        node_ids = ["robot"] + [f"frontier-{index}" for index in range(start, start + count)]
        return {"node_ids": node_ids, "cost_matrix": [[None] * len(node_ids) for _ in node_ids]}

    return {
        "schema_version": "2.0", "request_id": "episode:obs-3:req-1",
        "episode_id": "episode", "observation_id": 3, "timestamp_ms": 1,
        "target_object": "mug", "robot_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "map": {"frame_id": "world", "revision": 3, "observation_id": 3},
        "rooms": [], "objects": objects,
        "semantic_matches": [], "frontiers": frontiers,
        "candidate_evidence": candidate_evidence,
        "perception": {"object_detection": {}, "image_text_match": {}},
        "navigation_history": {},
        "original_policy": {
            "active_frontier_path_graph": graph(0, active_count),
            "dormant_frontier_path_graph": graph(active_count, dormant_count),
        },
        "search_branches": [
            {"name": name, "entity_results": []}
            for name in (
                "high_confidence_object_normal", "current_over_depth_object",
                "suspicious_object_normal", "active_frontier_policy",
                "dormant_frontier_policy", "original_object_extreme",
                "cached_over_depth_extreme",
            )
        ] + [{
            "name": "relaxed_all_cells_extreme",
            "entity_results": ([{"entity_ids": [item["id"] for item in objects]}] if objects else []),
        }],
        "observation_actions": [],
        "candidates": [{
            "id": "frontier-0-goal", "kind": "explore_frontier", "target_id": "frontier-0",
            "goal_pose": {"x": 0.0, "y": 1.0, "yaw": 1.57},
            "path": {"reachable": True, "collision_free": True, "distance_m": 1.0},
            "semantic_score": -0.2, "information_gain": None,
        }],
        "coverage": {
            "scope": "test",
            "entity_counts": {
                "objects": {"total": object_count, "exported": object_count, "omitted": 0},
                "active_frontiers": {"total": active_count, "exported": active_count, "omitted": 0},
                "dormant_frontiers": {"total": dormant_count, "exported": dormant_count, "omitted": 0},
                "candidate_evidence": {
                    "total": len(candidate_evidence), "exported": len(candidate_evidence), "omitted": 0
                },
                "candidates": {"total": 1, "exported": 1, "omitted": 0},
            },
            "required_feature_status": {
                "entity_enumeration": "complete", "normal_path_searches": "complete",
                "extreme_search_evidence": "complete_but_not_selectable_for_safety",
                "frontier_path_graphs": "complete_for_normally_reachable_frontiers",
                "raw_decision_neighborhoods": "complete", "entire_raw_map": "not_exported",
            },
            "high_level_evidence_complete": True, "raw_map_complete": False,
            "limitations": ["whole raw map not exported"],
        },
    }


class FixedProvider:
    def choose_request(self, request):
        self.request = request
        return "frontier-0-goal", 0.01, {"frontier-0-goal": 0.01}


class InformationParityContractTests(unittest.TestCase):
    def test_full_schema_has_no_legacy_entity_caps(self):
        prepared = prepare_decision(schema2_snapshot())
        self.assertEqual(len(prepared.state["objects"]), 70)
        self.assertEqual(len(prepared.state["frontiers"]), 15)
        self.assertTrue(prepared.state["coverage"]["high_level_evidence_complete"])

    def test_coverage_must_match_exported_arrays(self):
        snapshot = schema2_snapshot()
        snapshot["coverage"]["entity_counts"]["objects"] = {"total": 70, "exported": 69, "omitted": 1}
        with self.assertRaisesRegex(SnapshotError, "does not match objects"):
            validate_snapshot(snapshot)

    def test_complete_context_rejects_missing_production_evidence(self):
        snapshot = schema2_snapshot(object_count=1, active_count=1, dormant_count=0)
        snapshot["objects"][0].pop("normal_path_search")
        with self.assertRaisesRegex(SnapshotError, "normal_path_search"):
            validate_snapshot(snapshot)

    def test_complete_context_rejects_mismatched_observation(self):
        snapshot = schema2_snapshot(object_count=1, active_count=1, dormant_count=0)
        snapshot["map"]["observation_id"] = 999
        with self.assertRaisesRegex(SnapshotError, "observation_id"):
            validate_snapshot(snapshot)

    def test_low_provider_confidence_executes_valid_choice(self):
        provider = FixedProvider()
        decision = decide(schema2_snapshot(), provider)
        self.assertEqual(decision["status"], "GOAL")
        self.assertEqual(decision["confidence"], 0.01)
        self.assertIsNone(provider.request["state"]["candidates"][0]["information_gain"])
        self.assertEqual(provider.request["state"]["candidates"][0]["semantic_score"], -0.2)

    def test_unreachable_candidate_stays_in_state_but_not_criteria(self):
        snapshot = schema2_snapshot()
        excluded = copy.deepcopy(snapshot["candidates"][0])
        excluded.update({"id": "frontier-1-goal", "target_id": "frontier-1"})
        excluded["path"].update({"reachable": False, "collision_free": False})
        snapshot["candidates"].append(excluded)
        snapshot["coverage"]["entity_counts"]["candidates"] = {"total": 2, "exported": 2, "omitted": 0}
        prepared = prepare_decision(snapshot)
        self.assertEqual(len(prepared.state["candidates"]), 2)
        self.assertNotIn("frontier-1-goal", prepared.options)

    def test_legacy_full_profile_is_explicitly_incomplete(self):
        legacy = schema2_snapshot(object_count=1, active_count=1, dormant_count=0)
        legacy["schema_version"] = "1.0"
        for key in ("coverage", "request_id", "observation_id"):
            legacy.pop(key)
        self.assertFalse(build_model_state(legacy, "full")["coverage"]["high_level_evidence_complete"])

    def test_invalid_provider_response_is_retryable(self):
        class BadProvider:
            def choose_request(self, request):
                return "hold", 1.0, {"hold": 1.0}
        with self.assertRaises(ProviderError) as caught:
            decide(schema2_snapshot(), BadProvider())
        self.assertTrue(caught.exception.retryable)


if __name__ == "__main__":
    unittest.main()
