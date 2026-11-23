"""Only a completed, consistent eight-rank step is counted as a local hit."""

import importlib.util
import unittest
from pathlib import Path


FILE = Path(__file__).resolve().parents[3] / "examples/recipes/glm53_dp8/local_dp_sync/count_hits.py"
SPEC = importlib.util.spec_from_file_location("dp_sync_hits", FILE)
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)


def step(outcomes):
    return [
        f"GLM53 local DP step rank={rank} sequence=32 owner=1 request=req position=160 outcome={outcome}"
        for rank, outcome in enumerate(outcomes)
    ]


class CoverageTest(unittest.TestCase):
    def test_all_eight_are_required(self):
        self.assertEqual(parser.count(step(["local"] * 8))["steps"]["all_8_local"], 1)
        for outcomes in (["local"] * 7, ["local"] * 7 + ["gloo"]):
            out = parser.count(step(outcomes))
            self.assertFalse(out["steps"])
            self.assertEqual(len(out["incomplete_or_disagree"]), 1)

    def test_observation_does_not_count_as_a_local_transfer(self):
        out = parser.count(step(["gloo_candidate"] * 8))
        self.assertEqual(out["steps"]["all_8_observed_candidate_rows"], 1)
        self.assertNotIn("all_8_local", out["steps"])

    def test_duplicate_rank_is_rejected(self):
        lines = step(["local"] * 8)
        with self.assertRaises(ValueError):
            parser.count([*lines, lines[0]])


if __name__ == "__main__":
    unittest.main()
