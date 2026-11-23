"""CPU checks for the pinned v0.30 overlay installer."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RECIPE = Path(__file__).resolve().parents[3] / "examples/recipes/glm53_dp8"
SPEC = importlib.util.spec_from_file_location("dp8_recipe", RECIPE / "prepare.py")
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class OverlayTest(unittest.TestCase):
    def test_source_guard_and_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "source" / "vllm_ascend" / "worker"
            base.mkdir(parents=True)
            (base / "worker.py").write_text("old")
            fixture = root / "fixture"
            workers = fixture / "worker"
            workers.mkdir(parents=True)
            (workers / "worker.py").write_text("new")
            (fixture / "base.json").write_text(json.dumps({"worker.py": hashlib.sha256(b"old").hexdigest()}))
            with patch.object(recipe, "RECIPE", fixture):
                source = base.parents[1]
                result = recipe.prepare(source, root / "out")
                self.assertEqual((result / "vllm_ascend/worker/worker.py").read_text(), "new")
                self.assertEqual((base / "worker.py").read_text(), "old")
                with self.assertRaises(FileExistsError):
                    recipe.prepare(source, result)
                (base / "worker.py").write_text("foreign")
                with self.assertRaises(ValueError):
                    recipe.prepare(source, root / "bad")
                self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
