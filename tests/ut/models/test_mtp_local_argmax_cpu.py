"""CPU checks for native MTP norm/head ownership and greedy shard reduction."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


class NativeMtpArgmaxTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[3]

    def test_native_head_uses_same_norm_and_wraps_step(self):
        method = load_method(self.repo / 'vllm_ascend/models/deepseek_mtp.py',
                             'AscendDeepSeekMTP', 'get_top_tokens', {'torch': torch})
        for step in (0, 1, 3):
            with self.subTest(step=step):
                hidden = torch.tensor([[2., -3., 1., 4.]])
                weights = torch.tensor([[1., 2., 3., 4.], [-2., 4., 5., -1.]])
                heads = []
                for index in range(2):
                    norm = torch.nn.LayerNorm(4)
                    norm.weight.data.copy_(torch.tensor([1., 3., 2., 0.5]) + index)
                    shared = Mock(side_effect=norm)
                    shared.head = weights + index
                    heads.append(shared)
                processor = SimpleNamespace(get_top_tokens=Mock(
                    side_effect=lambda head, states: (states @ head.T).argmax(-1)))
                model = SimpleNamespace(num_mtp_layers=2, mtp_start_layer_idx=78,
                    layers={str(78 + i): SimpleNamespace(shared_head=h) for i, h in enumerate(heads)},
                    logits_processor=processor)
                actual = method(SimpleNamespace(model=model), hidden, step)
                selected = heads[step % 2]
                selected.assert_called_once_with(hidden)
                heads[1 - step % 2].assert_not_called()
                normed = processor.get_top_tokens.call_args.args[1]
                self.assertIs(processor.get_top_tokens.call_args.args[0], selected.head)
                torch.testing.assert_close(actual, (normed @ selected.head.T).argmax(-1), rtol=0, atol=0)

    def test_proposer_routes_existing_opt_in_to_distributed_greedy(self):
        method = load_method(self.repo / 'vllm_ascend/spec_decode/llm_base_proposer.py',
            'AscendSpecDecodeBaseProposer', 'compute_draft_token_ids',
            {'torch': torch, 'SamplingMetadata': object})
        hidden = torch.randn(2, 4)
        ids = torch.tensor([154879, 19360])
        sampler = Mock(return_value=ids)
        model = SimpleNamespace(compute_logits=Mock(side_effect=AssertionError('full logits used')))
        proposer = SimpleNamespace(method='mtp', use_local_argmax_reduction=True,
                                   _greedy_sample=sampler, model=model)
        result, probs = method(proposer, hidden, None)
        self.assertIs(result, ids)
        self.assertIsNone(probs)
        sampler.assert_called_once_with(hidden)
        model.compute_logits.assert_not_called()

    def test_default_mtp_keeps_full_logits_and_probability_path(self):
        method = load_method(self.repo / 'vllm_ascend/spec_decode/llm_base_proposer.py',
            'AscendSpecDecodeBaseProposer', 'compute_draft_token_ids',
            {'torch': torch, 'SamplingMetadata': object})
        hidden = torch.randn(2, 4)
        logits, metadata, probs = torch.randn(2, 8), object(), torch.ones(2, 8) / 8
        expected = torch.tensor([2, 3])
        sample = Mock(return_value=(expected, probs))
        proposer = SimpleNamespace(use_local_argmax_reduction=False, method='mtp',
            model=SimpleNamespace(compute_logits=Mock(return_value=logits)),
            _sample_draft_from_logits=sample)
        ids, actual_probs = method(proposer, hidden, metadata)
        self.assertIs(ids, expected)
        self.assertIs(actual_probs, probs)
        sample.assert_called_once_with(logits, metadata)

    def test_first_and_recurrent_mtp_branches_reach_distributed_greedy(self):
        path = self.repo / 'vllm_ascend/spec_decode/llm_base_proposer.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendSpecDecodeBaseProposer')
        merged = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_run_merged_draft')
        branches = [n for n in ast.walk(merged) if isinstance(n, ast.If)
                    and 'use_local_argmax_reduction' in ast.unparse(n.test)
                    and 'self.method' in ast.unparse(n.test)]
        self.assertEqual(len(branches), 2, 'Both initial and recurrent MTP steps must route the option')
        helper = load_method(path, 'AscendSpecDecodeBaseProposer', 'compute_draft_token_ids',
                             {'torch': torch, 'SamplingMetadata': object})
        for branch in branches:
            with self.subTest(condition=ast.unparse(branch.test)):
                step0 = any(isinstance(n, ast.Name) and n.id == 'draft_probs_step0'
                            for statement in branch.body for n in ast.walk(statement))
                function = ast.parse('def run(self, sample_hidden_states, sampling_metadata):\n pass').body[0]
                function.body = branch.body + ast.parse(
                    'return draft_token_ids, ' + ('draft_probs_step0' if step0 else 'draft_probs_step')).body
                namespace = {}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                             str(path), 'exec'), namespace)
                hidden, ids = torch.randn(2, 4), torch.tensor([154879, 19360])
                proposer = SimpleNamespace(use_local_argmax_reduction=True, method='mtp',
                    _greedy_sample=Mock(return_value=ids))
                proposer.compute_draft_token_ids = lambda h, m: helper(proposer, h, m)
                result, probs = namespace['run'](proposer, hidden, None)
                self.assertIs(result, ids)
                self.assertIsNone(probs)
                proposer._greedy_sample.assert_called_once_with(hidden)
                condition = compile(ast.Expression(branch.test), str(path), 'eval')
                self.assertTrue(eval(condition, {}, {'self': proposer}))
                proposer.use_local_argmax_reduction = False
                self.assertFalse(eval(condition, {}, {'self': proposer}))
                proposer.use_local_argmax_reduction, proposer.method = True, 'dspark'
                self.assertFalse(eval(condition, {}, {'self': proposer}))

    def test_upstream_sharded_argmax_matches_full_vocab_ties_padding_and_large_ids(self):
        import vllm
        path = Path(vllm.__file__).parent / 'model_executor/layers/logits_processor.py'
        vocab, shard_size, tp = 154880, 19392, 8
        logits = torch.full((3, shard_size * tp), -20.)
        logits[0, 154879] = 7.
        logits[1, 2 * shard_size + 4] = 8.
        logits[1, 5 * shard_size + 1] = 8.
        logits[2, 0] = 9.
        logits[:, vocab:] = 1000.  # Padding must never win.
        expected = logits[:, :vocab].argmax(-1)
        pairs = []
        for rank in range(tp):
            local = logits[:, rank * shard_size:(rank + 1) * shard_size].clone()
            padding = max(0, (rank + 1) * shard_size - vocab)
            if padding:
                local[:, -padding:] = -float('inf')
            values, indices = local.max(-1)
            pairs.append(torch.stack([values.float(), (indices + rank * shard_size).float()], -1))
        for rank in range(tp):
            with self.subTest(rank=rank):
                gather = Mock(side_effect=lambda pair, dim: torch.cat(pairs, -1))
                method = load_method(path, 'LogitsProcessor', 'get_top_tokens',
                    {'torch': torch, 'VocabParallelEmbedding': object,
                     'tensor_model_parallel_all_gather': gather})
                head = SimpleNamespace(tp_size=tp, shard_indices=SimpleNamespace(
                    num_org_vocab_padding=max(0, (rank + 1) * shard_size - vocab),
                    org_vocab_start_index=rank * shard_size))
                processor = SimpleNamespace(scale=1., soft_cap=None, _apply_head=lambda *args:
                    logits[:, rank * shard_size:(rank + 1) * shard_size].clone())
                actual = method(processor, head, torch.zeros(3, 4))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                pair = gather.call_args.args[0]
                torch.testing.assert_close(pair, pairs[rank], rtol=0, atol=0)
                self.assertEqual(pair.shape, (3, 2))


if __name__ == '__main__':
    unittest.main()
