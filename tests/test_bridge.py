import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from nav_jev_bridge.bridge import DemoProvider, OpenRouterJevProvider, ProviderError, decide


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "snapshot.json"


class Provider:
    def __init__(self, choice, confidence=0.9, probability=0.9):
        self.choice = choice
        self.confidence = confidence
        self.probability = probability

    def choose(self, state, options):
        return self.choice, self.confidence, {self.choice: self.probability}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_demo_selects_valid_candidate(self):
        result = decide(self.snapshot, DemoProvider())
        self.assertEqual(result["status"], "GOAL")
        self.assertIn(result["candidate_id"], {c["id"] for c in self.snapshot["candidates"]})

    def test_unreachable_candidate_never_reaches_provider(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["candidates"][0]["path"]["reachable"] = False

        class InspectProvider:
            def choose(self, state, options):
                self_options = options
                assert "approach-mug" not in self_options
                return "approach-mug", 1.0, {"approach-mug": 1.0}

        with self.assertRaisesRegex(ProviderError, "invalid selected choice"):
            decide(snapshot, InspectProvider())

    def test_provider_confidence_does_not_replace_perception_evidence(self):
        result = decide(self.snapshot, Provider("approach-mug", confidence=0.1))
        self.assertEqual(result["status"], "GOAL")

    def test_frontier_exploration_does_not_require_target_confidence(self):
        result = decide(
            self.snapshot,
            Provider("explore-living", confidence=0.1, probability=0.1),
        )
        self.assertEqual(result["status"], "GOAL")
        self.assertEqual(result["kind"], "explore_frontier")

    def test_hold_is_not_offered_when_an_executable_candidate_exists(self):
        class InspectProvider:
            def choose(self, state, options):
                assert "hold" not in options
                return "explore-living", 0.8, {"explore-living": 0.8}

        self.assertEqual(decide(self.snapshot, InspectProvider())["status"], "GOAL")

    def test_provider_cannot_hold_when_an_executable_candidate_exists(self):
        with self.assertRaisesRegex(ProviderError, "invalid selected choice"):
            decide(self.snapshot, Provider("hold"))

    def test_observation_candidate_is_executable(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["observation_actions"] = [
            {"id": "observe-left", "direction": "left", "purpose": "acquire observation"}
        ]
        snapshot["candidates"].append({
            "id": "observe-left-goal",
            "kind": "observe_rotation",
            "target_id": "observe-left",
            "goal_pose": dict(snapshot["robot_pose"]),
            "path": {"reachable": True, "collision_free": True, "distance_m": 0.0},
            "semantic_score": 0.0,
            "information_gain": 1.0,
        })

        class InspectProvider:
            def choose(self, state, options):
                assert "direction=left" in options["observe-left-goal"]
                return "observe-left-goal", 0.2, {"observe-left-goal": 0.2}

        result = decide(snapshot, InspectProvider())
        self.assertEqual(result["status"], "GOAL")
        self.assertEqual(result["kind"], "observe_rotation")

    def test_invalid_coordinates_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["candidates"][0]["goal_pose"]["x"] = float("nan")
        with self.assertRaises(ValueError):
            decide(snapshot, DemoProvider())

    def test_candidate_must_reference_observed_entity(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["candidates"][0]["target_id"] = "missing-object"
        with self.assertRaises(ValueError):
            decide(snapshot, DemoProvider())

    def test_openrouter_decisions_request_shape(self):
        seen = {}
        self.snapshot["map"]["local_grid"] = {
            "width": 1, "height": 1, "cell_size_m": 0.5,
            "occupancy_rows": ["?"], "semantic_value_rows": ["."],
        }
        self.snapshot["objects"][1]["class_scores"] = [
            {"class_index": 0, "label": "mug", "confidence": 0.83, "observations": 4}
        ]

        def fake_urlopen(request, timeout):
            seen["url"] = request.full_url
            seen["authorization"] = request.get_header("Authorization")
            seen["payload"] = json.loads(request.data)
            seen["timeout"] = timeout
            return io.BytesIO(json.dumps({
                "answers": {
                    "next_goal": {
                        "choice": "explore-living",
                        "confidence": 0.9,
                        "probabilities": {"explore-living": 0.9, "hold": 0.1},
                    }
                }
            }).encode())

        with patch("nav_jev_bridge.bridge.urllib.request.urlopen", side_effect=fake_urlopen):
            result = decide(self.snapshot, OpenRouterJevProvider(api_key="test-key", timeout_s=3.0))

        self.assertEqual(result["status"], "GOAL")
        self.assertEqual(seen["url"], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(seen["authorization"], "Bearer test-key")
        self.assertEqual(seen["payload"]["model"], "typesafe/jev-1.13")
        self.assertEqual(seen["payload"]["questions"]["next_goal"]["type"], "choice")
        self.assertNotIn("hold", seen["payload"]["questions"]["next_goal"]["criteria"])
        self.assertIn("weak target evidence is expected", seen["payload"]["questions"]["next_goal"]["instructions"])
        self.assertEqual(seen["payload"]["state"]["map"]["local_grid"]["occupancy_rows"], ["?"])
        self.assertEqual(seen["payload"]["state"]["objects"][1]["class_scores"][0]["confidence"], 0.83)
        self.assertEqual(seen["timeout"], 3.0)

    def test_choice_criteria_include_observed_evidence(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["objects"][1]["target_confidence"] = 0.83
        snapshot["objects"][1]["target_observations"] = 4
        snapshot["frontiers"][0]["semantic_confidence"] = 0.7
        snapshot["current_view"] = {
            "raw_score": 0.21,
            "score_type": "blip2_itc_cosine_similarity",
            "valid": True,
            "backend": "blip2_itm",
        }
        snapshot["navigation_history"] = {
            "last_entity_id": "object-mug",
            "last_candidate_id": "approach-mug",
            "consecutive_same_entity_selections": 3,
            "last_goal_execution": {
                "last_step_progress_m": 0.0,
                "cumulative_progress_m": 0.4,
            },
        }

        class InspectProvider:
            def choose(self, state, options):
                assert state["objects"][1]["target_confidence"] == 0.83
                summary = state["decision_summary"]
                assert summary["perception"]["image_text_match"]["raw_score"] == 0.21
                assert summary["eligible_candidate_count"] == len(snapshot["candidates"])
                mug_summary = next(
                    item for item in summary["eligible_candidates"]
                    if item["id"] == "approach-mug"
                )
                assert mug_summary["same_entity_as_last_choice"] is True
                assert mug_summary["consecutive_same_entity_selections"] == 3
                assert mug_summary["last_step_progress_m"] == 0.0
                assert "target_confidence=0.83" in options["approach-mug"]
                assert "target_observations=4" in options["approach-mug"]
                assert "current_view_raw_score=0.21" in options["approach-mug"]
                assert "repeat_count=3" in options["approach-mug"]
                assert "semantic_fusion_weight=0.7" in options["explore-living"]
                return "approach-mug", 0.9, {"approach-mug": 0.9}

        self.assertEqual(decide(snapshot, InspectProvider())["status"], "GOAL")

    def test_state_profiles_control_information_sent_to_provider(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["room_prior"] = "kitchen"
        snapshot["target_subcategories"] = ["coffee mug"]
        snapshot["map"]["local_grid"] = {"occupancy_rows": ["?.#"]}
        snapshot["objects"][1]["target_confidence"] = 0.83
        snapshot["objects"][1]["class_scores"] = [{"label": "mug", "confidence": 0.83}]

        class InspectProvider:
            def __init__(self, profile):
                self.profile = profile

            def choose(self, state, options):
                if self.profile == "candidates":
                    assert "local_grid" not in state["map"]
                    assert "room_prior" not in state
                    assert "target_confidence" not in state["objects"][1]
                    assert "target_confidence=unknown" in options["approach-mug"]
                elif self.profile == "spatial":
                    assert state["map"]["local_grid"]["occupancy_rows"] == ["?.#"]
                    assert state["objects"][1]["target_confidence"] == 0.83
                    assert "class_scores" not in state["objects"][1]
                    assert "room_prior" not in state
                else:
                    assert state["room_prior"] == "kitchen"
                    assert state["target_subcategories"] == ["coffee mug"]
                    assert state["objects"][1]["class_scores"][0]["confidence"] == 0.83
                return "approach-mug", 0.9, {"approach-mug": 0.9}

        for profile in ("candidates", "spatial", "full"):
            with self.subTest(profile=profile):
                self.assertEqual(
                    decide(snapshot, InspectProvider(profile), state_profile=profile)["status"],
                    "GOAL",
                )


if __name__ == "__main__":
    unittest.main()
