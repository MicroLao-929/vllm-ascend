"""CPU tests for the opt-in owner path and both native materialization points."""
import ast
import importlib.util
import logging
import sys
import textwrap
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import os

package = os.environ.get("GLM53_TEST_ASCEND")
if not package:
    if __name__ == "__main__":
        raise SystemExit("Set GLM53_TEST_ASCEND to the prepared --lazy-tables directory")
    raise unittest.SkipTest("GLM53_TEST_ASCEND is required for pinned runner tests")
WORKER = Path(package).resolve() / "vllm_ascend/worker"
lazy_spec = importlib.util.spec_from_file_location("glm53_mk_dp8_lazy_tables", WORKER / "glm53_mk_dp8_lazy_tables.py")
lazy = importlib.util.module_from_spec(lazy_spec)
lazy_spec.loader.exec_module(lazy)
MODULE = "vllm_ascend.worker.glm53_mk_dp8_lazy_tables"
sys.modules[MODULE] = lazy
spec = importlib.util.spec_from_file_location("tested_glm53_early_runner", WORKER / "glm53_mk_early_runner.py")
early = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = early
spec.loader.exec_module(early)


def states():
    rid, pos, graph = "request-A", 173, object()
    batch = NS(num_reqs=1, req_ids=[rid], num_computed_tokens_cpu=[pos], num_prompt_tokens=[128])
    runner = NS(
        input_batch=batch, max_num_reqs=1,
        parallel_config=NS(tensor_parallel_size=1, data_parallel_size=8, pipeline_parallel_size=1,
                           prefill_context_parallel_size=1, decode_context_parallel_size=1),
        model_config=NS(enforce_eager=True, is_encoder_decoder=False),
        use_async_scheduling=False, use_async_spec_decode=False, prepare_inputs_event=None,
        speculative_config=None, use_dcp=False, use_compress=False, dynamic_eplb=False,
        cache_config=NS(enable_prefix_caching=False, mamba_cache_mode="none"),
        device_metadata_executor=None, kvpp=NS(scheduler=None), long_seq_metadata=None,
        sparse_kv_offload_enabled=False, is_mm_prefix_lm=False, supports_mm_inputs=False,
    )
    hook = NS(
        early_runner=NS(rollout=NS(mode="on", min_shadow=8), ledger=NS(shadow_steps=8, shadow_clean=True)),
        native_builder_deferral="on", tp_size=1, plan=NS(size=8, depth=45), invalid=False,
        active=None, verified_graph=graph, bridge=NS(_poisoned=False, _graph=graph, _request=rid,
                                                   _position=pos - 1, max_context_tokens=2048),
        request_id=rid, next_position=pos, _last_completed_round=(rid, pos - 1),
    )
    output = NS(total_num_scheduled_tokens=1, num_scheduled_tokens={rid: 1}, scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={}, scheduled_new_reqs=[], finished_req_ids=set(),
                scheduled_cached_reqs=NS(resumed_req_ids=set()), preempted_req_ids=None,
                block_table_updates=None, kv_cache_block_copies=None, new_block_ids_to_zero=None,
                has_sync_kv_loads=False)
    return runner, hook, output


def func(tree, name):
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise AssertionError(f"ambiguous function {name} in production patch")
    return functions[0]


RUNNER = ast.parse((WORKER / "model_runner_v1.py").read_text(encoding="utf-8"))
HOOK = ast.parse((WORKER / "glm53_mk_ep.py").read_text(encoding="utf-8"))
PREP = func(RUNNER, "_prepare_inputs")
NATIVE_BUILD = func(RUNNER, "build_native_metadata")
RESTORE = func(RUNNER, "restore_native_attention_metadata")


class Tables:
    def __init__(self, cpu_page=91):
        self.events = []
        self.page = cpu_page
        self.device_page = None
        self.slot = None

    def commit_block_table(self, n):
        self.events.append(("commit", n))
        self.device_page = self.page

    def compute_slot_mapping(self, count, query, positions):
        self.events.append(("map", count, tuple(query), tuple(positions)))
        self.slot = self.device_page * 128 + positions[0] % 128


def exec_source(src, name, env=None):
    scope = {} if env is None else dict(env)
    exec(src, scope)
    return scope[name]


def prepare(tables, pending, pos=173):
    # Execute both exact operations from the installed runner, without importing NPU libraries.
    stmts = [n for n in PREP.body if isinstance(n, ast.If) and
             ast.unparse(n.test) == "glm53_deferred_tables is None"]
    if len(stmts) != 2:
        raise AssertionError("expected one guarded commit and one guarded slot mapping")
    run = "def publish(self, glm53_deferred_tables, num_reqs, total_num_scheduled_tokens):\n"
    run += textwrap.indent("\n".join(ast.unparse(n) for n in stmts), "    ")
    runner = NS(input_batch=NS(block_table=tables), query_start_loc=NS(gpu=[0, 1, -1]), positions=[pos, -99])
    exec_source(run, "publish")(runner, pending, 1, 1)


def native_functions(tables, pending):
    # Keep production metadata arguments, the actual one-shot builder, and binding to the live context.
    expected = {"device_private_slot": tables}
    common = object()
    calls = []
    def builder(**kwargs):
        calls.append(tables.slot)
        if tables.slot is None:
            raise AssertionError("native builder read uninitialized slot mapping")
        return expected, common
    runner = NS(_build_attention_metadata=builder, synchronize_input_prep=nullcontext)
    context = NS(attn_metadata=None, glm53_cpu_addresses=("old",))
    setup = """def make(self, lazy_tables, early_context, deferred):
    num_tokens_unpadded = num_tokens_padded = num_reqs = num_reqs_padded = max_num_scheduled_tokens = 1
    ubatch_slices_attn = logits_indices = use_spec_decode = cascade_attn_prefix_lens = None
    cudagraph_mode = batch_desc = None
    scheduler_output = type('Step', (), {'num_scheduled_tokens': {'request-A': 1}})()
    num_scheduled_tokens_np = [1]
    attn_metadata = spec_decode_common_attn_metadata = None
"""
    setup += textwrap.indent(ast.unparse(NATIVE_BUILD), "    ") + "\n"
    setup += "    early_prepared = deferred(build_native_metadata)\n"
    setup += textwrap.indent(ast.unparse(RESTORE), "    ") + "\n"
    setup += "    def metadata():\n        return attn_metadata, spec_decode_common_attn_metadata\n"
    setup += "    return build_native_metadata, restore_native_attention_metadata, metadata\n"
    build, restore, metadata = exec_source(setup, "make")(runner, pending, context, early.DeferredNativeMetadata)
    return NS(context=context, build=build, restore=restore, metadata=metadata, calls=calls,
              expected=expected, common=common)


class AdmissionTests(unittest.TestCase):
    def test_128_in_128_out_continuation(self):
        self.assertTrue(lazy.eligible(*states(), deferred_corrections=None))

    def test_native_prefill_warmup_boundaries_and_missing_schema(self):
        for pos, allowed in ((128, False), (129, True), (255, True), (2048, False)):
            with self.subTest(pos=pos):
                r, h, o = states()
                r.input_batch.num_computed_tokens_cpu[0] = h.next_position = pos
                h.bridge._position = pos - 1
                h._last_completed_round = (h.request_id, pos - 1)
                self.assertEqual(lazy.eligible(r, h, o, None), allowed)
        for attr in ("new_block_ids_to_zero", "has_sync_kv_loads"):
            r, h, o = states()
            delattr(o, attr)
            self.assertFalse(lazy.eligible(r, h, o, None))

    def test_unknown_interrupted_unsafe_and_other_topologies_use_native(self):
        cases = [
            ("runner.parallel_config.tensor_parallel_size", 8), ("runner.parallel_config.data_parallel_size", 1),
            ("runner.parallel_config.pipeline_parallel_size", 2), ("runner.parallel_config.prefill_context_parallel_size", 2),
            ("runner.parallel_config.decode_context_parallel_size", 2), ("runner.input_batch.num_reqs", 0),
            ("runner.input_batch.num_reqs", 2), ("runner.speculative_config", object()),
            ("runner.use_async_scheduling", True), ("runner.prepare_inputs_event", object()),
            ("runner.cache_config.enable_prefix_caching", True), ("runner.cache_config.mamba_cache_mode", "align"),
            ("runner.device_metadata_executor", object()), ("runner.sparse_kv_offload_enabled", True),
            ("runner.model_config.enforce_eager", False), ("runner.kvpp.scheduler", object()),
            ("hook.plan.depth", 1), ("hook.invalid", True), ("hook.bridge._graph", object()),
            ("hook.early_runner.rollout.mode", "shadow"), ("hook.native_builder_deferral", "off"),
            ("hook.early_runner.ledger.shadow_steps", 7), ("hook.early_runner.ledger.shadow_clean", False),
            ("hook._last_completed_round", ("other", 172)), ("hook.bridge._position", 171),
            ("hook.request_id", "other"), ("hook.active", object()), ("hook.verified_graph", None),
            ("output.total_num_scheduled_tokens", 2), ("output.num_scheduled_tokens", {"other": 1}),
            ("output.scheduled_cached_reqs.resumed_req_ids", {"request-A"}), ("output.preempted_req_ids", {"x"}),
            ("output.scheduled_new_reqs", [object()]), ("output.scheduled_spec_decode_tokens", {"x": [1]}),
            ("output.block_table_updates", {"request-A": ([12],)}), ("output.kv_cache_block_copies", [object()]),
            ("output.new_block_ids_to_zero", [88]), ("output.has_sync_kv_loads", True),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                r, h, o = states()
                parts = path.split(".")
                target = {"runner": r, "hook": h, "output": o}[parts[0]]
                for field in parts[1:-1]:
                    target = getattr(target, field)
                setattr(target, parts[-1], value)
                self.assertFalse(lazy.eligible(r, h, o, None))
        self.assertFalse(lazy.eligible(*states(), deferred_corrections=object()))


class PublicationTests(unittest.TestCase):
    def test_normal_runner_publishes_native_tables(self):
        tables = Tables()
        prepare(tables, None)
        self.assertEqual(tables.events, [("commit", 1), ("map", 1, (0, 1), (173,))])
        self.assertEqual(tables.slot, 91 * 128 + 45)

    def test_successful_mk_never_publishes_native_tables(self):
        tables = Tables()
        pending = lazy.DeferredBlockTables(tables)
        prepare(tables, pending)
        self.assertEqual(tables.events, [])
        self.assertFalse(pending.restored)

    def test_early_decline_native_builder_publishes_in_original_order(self):
        tables = Tables(229)
        pending = lazy.DeferredBlockTables(tables)
        prepare(tables, pending)
        f = native_functions(tables, pending)
        self.assertEqual(f.build(), (f.expected, f.common))
        self.assertEqual(f.calls, [229 * 128 + 45])
        self.assertTrue(pending.restored)

    def test_late_vote_miss_sets_live_context_and_outer_outputs_once(self):
        tables = Tables(305)
        pending = lazy.DeferredBlockTables(tables)
        prepare(tables, pending)
        f = native_functions(tables, pending)
        f.context.glm53_materialize_native_attention_metadata = f.restore
        materialize = func(HOOK, "_materialize_early_metadata")
        forward = func(HOOK, "forward_round")
        host = NS(early_runner=None, tp_size=1, _vote=lambda *a: ("real", "request-A", 173),
                  gate=NS(sequence=9, rank=0, begin=lambda *a, **kw: None))
        host._materialize_early_metadata = MethodType(
            exec_source(ast.unparse(materialize), materialize.name), host)
        vote = exec_source(ast.unparse(forward), forward.name, {"logger": logging.getLogger(__name__)})
        self.assertIsNone(vote(host, None, None, False, f.context, 1, None, None))
        f.restore()
        self.assertIs(f.context.attn_metadata, f.expected)
        self.assertEqual(f.metadata(), (f.expected, f.common))
        self.assertIsNone(f.context.glm53_cpu_addresses)
        self.assertEqual(f.calls, [305 * 128 + 45])
        self.assertEqual(len(tables.events), 2)

    def test_failed_mapping_or_invalid_reentry_never_replays(self):
        tables = Tables()
        pending = lazy.DeferredBlockTables(tables)
        with self.assertRaises(RuntimeError):
            pending.materialize()
        with self.assertRaises(RuntimeError):
            pending.save_mapping(2, [0, 1, 2], [7, 8])
        prepare(tables, pending)
        with self.assertRaises(RuntimeError):
            prepare(tables, pending)
        def fail(*args):
            raise ValueError("device table rejected")
        tables.compute_slot_mapping = fail
        with self.assertRaisesRegex(ValueError, "device table rejected"):
            pending.materialize()
        with self.assertRaises(RuntimeError):
            pending.materialize()
        self.assertFalse(pending.restored)
        self.assertEqual(tables.events, [("commit", 1)])

    def test_new_round_republishes_own_page_and_token_position(self):
        tables = Tables(81)
        for page, pos in ((81, 175), (158, 176), (202, 255)):
            tables.page = page
            pending = lazy.DeferredBlockTables(tables)
            prepare(tables, pending, pos)
            native_functions(tables, pending).build()
            self.assertEqual(tables.slot, page * 128 + pos % 128)

    def test_geometry_guard_and_second_late_fallback_run_before_consumers(self):
        execute = ast.unparse(func(RUNNER, "execute_model"))
        guard = execute.index("if lazy_tables is not None and (cascade_attn_prefix_lens")
        self.assertLess(guard, execute.index("self._pad_query_start_loc_for_fia"))
        route = ast.unparse(func(HOOK, "forward_round"))
        rebind = route[route.index("if not self._early_plan_ready(runner):"):]
        self.assertLess(rebind.index("self._materialize_early_metadata(context)"), rebind.index("self.try_decode"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
