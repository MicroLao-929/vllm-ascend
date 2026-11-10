"""CPU-only bridge regressions; importing vLLM/NPU kernels is unnecessary."""
import __future__
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/glm_5_2_mega.py"
TREE = ast.parse(SOURCE.read_text())


def load_function(name):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


supports = load_function("_supports_mega_batch")


def metadata(lengths, *, starts=True, padded=None):
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    result = SimpleNamespace(num_actual_tokens=prefix[-1],
        seq_lens=torch.tensor([2056] * len(lengths)),
        block_table=torch.zeros((len(lengths), 1), dtype=torch.int32),
        slot_mapping=torch.arange(padded or prefix[-1]))
    if starts:
        result.query_start_loc = torch.tensor(prefix)
    else:
        result.cum_query_lens = torch.tensor(prefix[1:])
    return result


class MetadataTests(unittest.TestCase):
    def test_rectangular_starts_and_prefix_ends(self):
        for starts in (False, True):
            self.assertTrue(supports(metadata([8], starts=starts), 8))
            self.assertTrue(supports(metadata([4, 4], starts=starts), 8))
            self.assertFalse(supports(metadata([3, 5], starts=starts), 8))
            self.assertFalse(supports(metadata([3, 4], starts=starts), 7))

    def test_padding_uses_native_output_extent(self):
        self.assertFalse(supports(metadata([8], padded=10), 10))

    def test_missing_lengths_only_allows_one_token_per_request(self):
        self.assertTrue(supports(SimpleNamespace(num_actual_tokens=2, seq_lens=torch.ones(2)), 2))
        self.assertFalse(supports(SimpleNamespace(num_actual_tokens=8, seq_lens=torch.ones(1)), 8))

    def test_single_request_needs_no_cpu_transfer(self):
        ends = SimpleNamespace(cpu=Mock(side_effect=AssertionError("unexpected host sync")))
        self.assertTrue(supports(SimpleNamespace(num_actual_tokens=8, seq_lens=torch.ones(1), cum_query_lens=ends), 8))
        ends.cpu.assert_not_called()

    def test_host_query_starts_avoid_device_transfer(self):
        meta = metadata([4, 4])
        meta.query_start_loc_cpu = meta.query_start_loc
        meta.query_start_loc = SimpleNamespace(__getitem__=None)
        self.assertTrue(supports(meta, 8))

    def test_forward_routes_ragged_and_padded_to_native(self):
        class Native:
            def forward(self, ids, *args, **kwargs):
                return ("native", ids.shape[0])
        original = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "AscendGlm52MegaForCausalLM")
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        definition = ast.ClassDef(name="Bridge", bases=[ast.Name(id="Native", ctx=ast.Load())],
                                  keywords=[], body=[method], decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
        current = {}
        namespace = dict(Native=Native, _mega_enabled=lambda: True,
            get_forward_context=lambda: SimpleNamespace(attn_metadata=current["meta"]),
            _extract_dsa_metadata=lambda x: x,
            _has_usable_metadata=load_function("_has_usable_metadata"),
            _supports_mega_batch=supports)
        exec(compile(module, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
        bridge = namespace["Bridge"]()
        bridge._mega_state = SimpleNamespace(set_kv_rope_layout=Mock())
        bridge._get_mega_debug = lambda: None
        bridge._is_decode_step = lambda meta: True
        bridge._forward_mega = Mock(return_value="mega")
        for lengths, input_count in (([3, 5], 8), ([8], 10)):
            current["meta"] = metadata(lengths)
            result = bridge.forward(torch.arange(input_count), torch.arange(input_count))
            self.assertEqual(result, ("native", input_count))
        bridge._forward_mega.assert_not_called()
        current["meta"] = metadata([8])
        self.assertEqual(bridge.forward(torch.arange(8), torch.arange(8)), "mega")
        bridge._forward_mega.assert_called_once()

    def test_private_forward_copies_only_actual_tokens(self):
        original = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "AscendGlm52MegaForCausalLM")
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_mega")
        # Keep production decorators: they are CPU-safe and must disable inference views.
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        buffers = dict(input_ids=torch.full((10,), -1, dtype=torch.int64),
            positions=torch.full((10,), -1, dtype=torch.int64),
            slot_mapping=torch.full((10,), -1, dtype=torch.int64),
            block_tables=torch.zeros((1, 1), dtype=torch.int32),
            kv_seq_len=torch.zeros(1, dtype=torch.int32), q_seq_len=torch.zeros(1, dtype=torch.int32))
        state = SimpleNamespace(num_blocks=1, set_kv_rope_layout=Mock(),
            get_meta_buffers=lambda size: buffers, build_forward_batch_info=lambda *args: None,
            mega=SimpleNamespace(forward=lambda info, ids, *args: ids.clone()))
        model = SimpleNamespace(vllm_config=None, _mega_state=state, _ensure_mega=lambda config: state,
            _fill_rope_caches=lambda *args: None, _get_mega_debug=lambda: None)
        result = namespace["_forward_mega"](model, torch.arange(10), torch.arange(10), metadata([8], padded=10))
        self.assertEqual(result.tolist(), list(range(8)))
        self.assertEqual(buffers["input_ids"].tolist(), list(range(8)) + [-1, -1])
        self.assertEqual(buffers["positions"].tolist(), list(range(8)) + [-1, -1])
        self.assertEqual(buffers["q_seq_len"].tolist(), [8])


if __name__ == "__main__":
    unittest.main()
