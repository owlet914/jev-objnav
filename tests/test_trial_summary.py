import json
import tempfile
import unittest
from pathlib import Path

from nav_jev_bridge.bridge import DemoProvider
from tools.replay_jev_traces import replay
from tools.summarize_ovon_trials import parse_run


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
            self.assertEqual(result["fallback_decision_rate"], 0.5)
            self.assertEqual(result["median_jev_latency_ms"], 210)
            self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
