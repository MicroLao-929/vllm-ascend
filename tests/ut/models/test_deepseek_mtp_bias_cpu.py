"""CPU regression for checkpoint-optional MTP RMSNorm bias cleanup."""
import ast
import unittest
from pathlib import Path

import torch


class OptionalNormBiasTests(unittest.TestCase):
    def test_only_unloaded_optional_norm_parameters_are_removed(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_mtp.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDeepSeekMTP")
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_remove_unloaded_norm_biases"
        )
        namespace = {"nn": torch.nn}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        model = torch.nn.Module()
        for name, loaded in (("absent", False), ("loaded", True), ("untracked", None)):
            norm = torch.nn.Module()
            norm.variance_epsilon = 1e-5
            norm.bias_loaded = loaded
            norm.bias = torch.nn.Parameter(torch.ones(4))
            model.add_module(name, norm)
        model.linear = torch.nn.Linear(4, 4)
        loaded_bias = model.loaded.bias
        linear_bias = model.linear.bias
        namespace["_remove_unloaded_norm_biases"](model)
        self.assertIsNone(model.absent.bias)
        self.assertIs(model.loaded.bias, loaded_bias)
        self.assertIsNotNone(model.untracked.bias)
        self.assertIs(model.linear.bias, linear_bias)


if __name__ == "__main__":
    unittest.main()
