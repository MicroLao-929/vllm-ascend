"""Exercise the actual optional foreach writeback on CPU tensors."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


WORKER = Path(__file__).resolve().parents[3] / "examples/recipes/glm53_dp8/worker/glm53_mk_batched.py"
SPEC = importlib.util.spec_from_file_location("glm53_mk_batched", WORKER)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CacheCopyTest(unittest.TestCase):
    def test_casts_and_shape_fallback(self):
        dst = [torch.empty(3, dtype=torch.float32) for _ in range(2)]
        src = [torch.arange(3), -torch.arange(3)]
        self.assertTrue(module.copy_pairs(zip(dst, src)))
        for actual, expected in zip(dst, src):
            self.assertTrue(torch.equal(actual, expected.float()))
        self.assertFalse(module.copy_pairs([(dst[0], src[0])]))
        self.assertFalse(module.copy_pairs([(dst[0], src[0]), (dst[1], torch.arange(2))]))

    def test_partial_failure_can_be_replayed(self):
        dst = [torch.zeros(3), torch.zeros(3)]
        src = [torch.ones(3), 2 * torch.ones(3)]

        def fail_after_first(destinations, sources):
            destinations[0].copy_(sources[0])
            raise RuntimeError("simulated incomplete fused copy")

        with patch.object(torch, "_foreach_copy_", fail_after_first):
            self.assertFalse(module.copy_pairs(zip(dst, src)))
        for actual, expected in zip(dst, src):
            actual.copy_(expected)
        self.assertTrue(all(torch.equal(actual, expected) for actual, expected in zip(dst, src)))


if __name__ == "__main__":
    unittest.main()
