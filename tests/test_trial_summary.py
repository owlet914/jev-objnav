import json
import tempfile
import unittest
from pathlib import Path

from nav_jev_bridge.bridge import DemoProvider
from tools.replay_jev_traces import replay
from tools.summarize_ovon_trials import parse_run
from tools.summarize_jev_inputs import summarize
from tools.analyze_jev_failure_modes import analyze


class TrialSummaryTests(unittest.TestCase):
    def test_replay_uses_same_snapshot_for_each_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            trace_dir = Path(temp)
            snapshot = json.loads(
                (Path(__file__).resolve().parents[1] / "examples" / "snapshot.json").read_text(
                    encoding="utf-8"
                )
            )
            (trace_dir / "one.json").write_text(
                json.dumps({"snapshot": snapshot}), encoding="utf-8"
            )
            result = replay(trace_dir, DemoProvider(), 1, [0.4, 0.6])
            self.assertEqual(result["snapshots"], 1)
            self.assertEqual(set(result["profiles"]), {"candidates", "spatial", "full"})
            self.assertTrue(all(row["goal_choices"] == 1 for row in result["profiles"].values()))

    def test_counts_success_and_fallback_without_claiming_partial_run_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "full"
            (run / "traces").mkdir(parents=True)
            (run / "record.txt").write_text(
                "Scene ID: scene_a\nEpisode ID: 1\nsuccess or not: success\n"
                "Scene ID: scene_b\nEpisode ID: 2\nsuccess or not: timeout\n",
                encoding="utf-8",
            )
            (run / "continue.txt").write_text(
                "| Total SPL | 0.75 |\n", encoding="utf-8"
            )
            for index, (status, latency) in enumerate((("GOAL", 120), ("HOLD", 300))):
                (run / "traces" / f"{index}.json").write_text(
                    json.dumps({"latency_ms": latency, "decision": {"status": status}}),
                    encoding="utf-8",
                )
            result = parse_run(run, expected_episodes=24)
            self.assertEqual(result["episodes_recorded"], 2)
            self.assertEqual(result["success_rate"], 0.5)
            self.assertEqual(result["spl"], 0.375)
            self.assertEqual(result["non_goal_decision_rate"], 0.5)
            self.assertEqual(result["median_jev_latency_ms"], 210)
            self.assertFalse(result["complete"])

    def test_multiword_jev_failure_and_input_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "traces").mkdir()
            (run / "record.txt").write_text(
                "Scene ID: scene_a\nEpisode ID: 1\nsuccess or not: jev decision failure\n",
                encoding="utf-8",
            )
            snapshot = json.loads((Path(__file__).resolve().parents[1] / "examples" / "snapshot.json").read_text(encoding="utf-8"))
            (run / "traces" / "one.json").write_text(
                json.dumps({"snapshot": snapshot, "decision": {"status": "HOLD", "reason": "provider_selected_hold"}, "state_profile": "full", "latency_ms": 42}),
                encoding="utf-8",
            )
            self.assertEqual(parse_run(run)["failure_reasons"], {"jev decision failure": 1})
            report = summarize(run / "traces")
            self.assertEqual(report["trace_files"], 1)
            self.assertEqual(report["hold_reasons"], {"provider_selected_hold": 1})
            self.assertEqual(report["entity_counts"]["candidates"]["median"], 2)

    def test_input_summary_reads_live_current_view_raw_score(self):
        with tempfile.TemporaryDirectory() as temp:
            trace_dir = Path(temp)
            state = {
                "episode_id": "scene:1",
                "candidates": [],
                "objects": [{"target_confidence": 0.75, "target_observations": 3}],
                "frontiers": [],
                "semantic_matches": [],
                "rooms": [],
                "map": {},
                "current_view": {
                    "raw_score": 0.21,
                    "score_type": "blip2_itc_cosine_similarity",
                    "valid": True,
                    "backend": "blip2_itm",
                },
                "navigation_history": {"consecutive_same_entity_selections": 7},
            }
            (trace_dir / "one.json").write_text(
                json.dumps({
                    "model_request": {"state": state},
                    "decision": {"status": "GOAL", "kind": "approach_object", "candidate_id": "o"},
                    "state_profile": "full",
                }),
                encoding="utf-8",
            )

            report = summarize(trace_dir)
            self.assertEqual(report["image_text_match_scores"]["count"], 1)
            self.assertAlmostEqual(report["image_text_match_scores"]["median"], 0.21)
            self.assertEqual(report["current_view_validity"], {"True": 1})
            self.assertEqual(report["object_target_confidences"]["median"], 0.75)
            self.assertEqual(report["consecutive_same_entity_selections"]["max"], 7)

    def test_failure_analysis_reports_evidence_conflict_and_no_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            trace_dir = Path(temp)
            state = {
                "episode_id": "scene:1",
                "observation_id": 9,
                "current_view": {"raw_score": 0.12},
                "candidates": [{
                    "id": "object-0-goal",
                    "kind": "approach_object",
                    "target_id": "object-0",
                    "path": {"distance_m": 1.5},
                }],
                "objects": [{
                    "id": "object-0",
                    "target_confidence": 0.7,
                    "target_observations": 5,
                    "provenance": {"observation_id": 8},
                }],
                "navigation_history": {
                    "consecutive_same_entity_selections": 4,
                    "last_goal_execution": {"last_step_progress_m": 0.0},
                },
            }
            (trace_dir / "one.json").write_text(
                json.dumps({
                    "model_request": {"state": state},
                    "decision": {
                        "status": "GOAL",
                        "kind": "approach_object",
                        "candidate_id": "object-0-goal",
                    },
                }),
                encoding="utf-8",
            )

            report = analyze(trace_dir)
            self.assertEqual(report["weak_view_high_map_conflict_count"], 1)
            self.assertEqual(report["repeated_object_no_progress_count"], 1)
            self.assertEqual(report["selected_object_not_from_current_observation_count"], 1)


if __name__ == "__main__":
    unittest.main()
